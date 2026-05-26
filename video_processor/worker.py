import os
import cv2
import yaml
import time
import base64
import requests
import io
import urllib.request
from PIL import Image
from celery import Celery
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import numpy as np
import math

# --- 1. CONFIGURACIÓN Y DESCARGA DINÁMICA DE MODELO ---
with open("/app/config.yml", "r") as f:
    cfg = yaml.safe_load(f)

vp_cfg = cfg['video_processor']
MODEL_DIR = "/app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILENAME = vp_cfg['model_name']
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = os.getenv("YOLO_MODEL_URL", "")
FORCE_DOWNLOAD = os.getenv("FORCE_DOWNLOAD_MODEL", "false").lower() == "true"

if FORCE_DOWNLOAD or (not os.path.exists(MODEL_PATH) and MODEL_URL):
    print(f"📥 Descargando modelo YOLO desde {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modelo descargado exitosamente.")

# Inicializar YOLO de forma global para el worker
print(f"🧠 Cargando modelo YOLO: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH)
tracker_path = f"/app/{vp_cfg['tracker_config']}"

# Configuración de YOLO
confidence_threshold = vp_cfg['detector']['confidence_threshold']
iou_threshold = vp_cfg['detector']['iou_threshold']
imgsz = vp_cfg['detector']['imgsz']

# --- 2. CONFIGURACIÓN DE CELERY ---
celery_app = Celery(
    'video_worker',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
)

API_URL = os.getenv("API_URL", "http://api:8000")


# --- 3. FUNCIONES MATEMÁTICAS Y GEOMÉTRICAS ---
def calculate_sharpness(img_crop):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def bbox_distance(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left and y_bottom < y_top:
        return math.hypot(x_left - x_right, y_top - y_bottom)
    elif x_right < x_left:
        return float(x_left - x_right)
    elif y_bottom < y_top:
        return float(y_top - y_bottom)
    else:
        return 0.0
    
def interpolate_trajectory(trajectory_dict):
    frames = sorted(trajectory_dict.keys())
    if not frames: return {}
    
    min_f, max_f = frames[0], frames[-1]
    full_trajectory = {}
    
    known_x1 = [trajectory_dict[f][0] for f in frames]
    known_y1 = [trajectory_dict[f][1] for f in frames]
    known_x2 = [trajectory_dict[f][2] for f in frames]
    known_y2 = [trajectory_dict[f][3] for f in frames]
    
    for f in range(min_f, max_f + 1):
        x1 = np.interp(f, frames, known_x1)
        y1 = np.interp(f, frames, known_y1)
        x2 = np.interp(f, frames, known_x2)
        y2 = np.interp(f, frames, known_y2)
        full_trajectory[f] = [x1, y1, x2, y2]
        
    return full_trajectory

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2: return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm != 0 else 0.0

def calculate_visible_ratio(crop_bgr):
    """Calcula el porcentaje de píxeles del crop que NO son fondo negro puro."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    non_black_pixels = cv2.countNonZero(gray)
    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    return non_black_pixels / total_pixels if total_pixels > 0 else 0

def is_queued_behind(box_anchor, box_doubtful):
    """Determina si el producto dudoso está en la misma fila (hacia el fondo) que el ancla."""
    ax1, ay1, ax2, ay2 = box_anchor
    dx1, dy1, dx2, dy2 = box_doubtful
    
    overlap_x = max(0, min(ax2, dx2) - max(ax1, dx1))
    width_d = dx2 - dx1
    if width_d == 0: return False
    ratio_x = overlap_x / width_d
    
    is_behind = dy2 < ay2 # La base del dudoso está más arriba en la imagen
    
    return ratio_x > 0.60 and is_behind


# --- TAREA: IMAGEN ESTÁTICA ---
@celery_app.task(name="tasks.detect_bboxes", bind=True)
def detect_bboxes_task(self, image_path):
    print(f"🔍 Ejecutando Pipeline Híbrido en Imagen Estática: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "FAILED", "error": "No se pudo leer la imagen"}

    h_img, w_img = img.shape[:2]
    
    margin_pct = vp_cfg['geometry']['edge_margin_pct']
    margin_x = int(w_img * margin_pct)
    margin_y = int(h_img * margin_pct)
    
    dbscan_eps = vp_cfg.get('clustering', {}).get('dbscan_eps', 15.0)
    min_samples = vp_cfg.get('clustering', {}).get('min_samples', 2)
    deep_sim_threshold = vp_cfg.get('clustering', {}).get('deep_similarity_threshold', 0.75)

    results = yolo_model(img, verbose=False, imgsz=imgsz, conf=confidence_threshold, iou=iou_threshold, device='cuda')
    
    active_in_frame = []
    
    if results[0].boxes is not None and results[0].masks is not None:
        boxes_data = results[0].boxes.xyxy.cpu().numpy().astype(int)
        masks_xy = results[0].masks.xy 
        
        self.update_state(state='PROCESSING', meta={'message': f'Extrayendo {len(boxes_data)} máscaras individuales...'})
        
        for box, mask_pts in zip(boxes_data, masks_xy):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            is_near_edge = (x1 <= margin_x) or (y1 <= margin_y) or (x2 >= w_img - margin_x) or (y2 >= h_img - margin_y)
            if is_near_edge: continue
                
            if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                if len(mask_pts) > 0:
                    obj_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    pts = np.array(mask_pts, np.int32)
                    cv2.fillPoly(obj_mask, [pts], 255)
                    masked_img = cv2.bitwise_and(img, img, mask=obj_mask)
                    crop = masked_img[y1:y2, x1:x2]
                else:
                    crop = img[y1:y2, x1:x2]
                
                # 🚀 Calculamos ratio visible al vuelo para el motor híbrido
                vis_ratio = calculate_visible_ratio(crop)
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG", quality=95)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                payload = {"image_b64": img_b64, "bboxes": [[0, 0, crop.shape[1], crop.shape[0]]]}
                try:
                    res = requests.post(f"{API_URL}/api/v1/predict", json=payload)
                    task_id = res.json().get('task_id')
                    
                    active_in_frame.append({
                        "task_id": task_id,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "vis_ratio": vis_ratio,
                        "item": None 
                    })
                except Exception as e:
                    print(f"Error encolando a API: {e}")
                    
    if not active_in_frame:
        try: os.remove(image_path)
        except: pass
        return {"status": "SUCCESS", "detections": []}

    self.update_state(state='PROCESSING', meta={'message': 'Esperando resolución de Re-ID de la API...'})
    
    for item_data in active_in_frame:
        t_id = item_data["task_id"]
        while True:
            res_poll = requests.get(f"{API_URL}/api/v1/results/{t_id}")
            if res_poll.status_code == 200:
                data = res_poll.json()
                if data["status"] == "SUCCESS":
                    detections = data["result"].get("detections", [])
                    if detections:
                        pred = detections[0].get("prediction", {})
                        item_data["item"] = {
                            "sku": pred.get("sku", "Sin SKU"),
                            "name": pred.get("name", "Desconocido"),
                            "verified": pred.get("verified", False),
                            "was_originally_verified": pred.get("verified", False),
                            "embedding": pred.get("embedding", None),
                            "tid": t_id, 
                            "best_frame": 0
                        }
                    break
                elif data["status"] == "FAILED":
                    item_data["item"] = {
                        "sku": "ERROR", "name": "Fallo API", "verified": False, 
                        "was_originally_verified": False, "embedding": None, "tid": t_id, "best_frame": 0
                    }
                    break
            time.sleep(0.4)

    # --- 3. MOTOR HÍBRIDO (DBSCAN + Semántica Dinámica) ---
    n_items = len(active_in_frame)
    if n_items >= 2:
        dist_matrix = np.zeros((n_items, n_items))
        for i in range(n_items):
            for j in range(i+1, n_items):
                dist = bbox_distance(active_in_frame[i]["bbox"], active_in_frame[j]["bbox"])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='precomputed')
        labels = clusterer.fit_predict(dist_matrix)
        
        for cluster_id in set(labels):
            if cluster_id == -1: continue
            
            cluster_members = [active_in_frame[i] for i in range(n_items) if labels[i] == cluster_id]
            anchors = [m for m in cluster_members if m["item"]["was_originally_verified"]]
            
            if anchors:
                for member in cluster_members:
                    item_dict = member["item"]
                    
                    if not item_dict["verified"]:
                        sorted_anchors = sorted(anchors, key=lambda a: (
                            bbox_distance(member["bbox"], a["bbox"]),
                            math.hypot( (member["bbox"][0]+member["bbox"][2])/2 - (a["bbox"][0]+a["bbox"][2])/2, 
                                         (member["bbox"][1]+member["bbox"][3])/2 - (a["bbox"][1]+a["bbox"][3])/2)
                        ))
                        
                        emb_dudoso = item_dict.get("embedding")
                        vis_ratio = member.get("vis_ratio", 1.0)
                        rescued = False
                        
                        for a in sorted_anchors:
                            emb_ancla = a["item"].get("embedding")
                            sim_score = cosine_similarity(emb_dudoso, emb_ancla)
                            
                            # 🚀 UMBRAL DINÁMICO
                            is_behind = is_queued_behind(a["bbox"], member["bbox"])
                            
                            if is_behind and vis_ratio < 0.40:
                                dyn_thresh = 0.55
                            elif is_behind:
                                dyn_thresh = 0.65
                            elif vis_ratio < 0.40:
                                dyn_thresh = 0.70
                            else:
                                dyn_thresh = deep_sim_threshold
                            
                            if sim_score >= dyn_thresh:
                                item_dict["verified"] = True
                                item_dict["sku"] = a["item"]["sku"]
                                item_dict["name"] = f"{a['item']['name']} (Contexto Híbrido)"
                                rescued = True
                                break

    # --- 4. EMPAQUETADO FINAL ---
    final_results = []
    for m in active_in_frame:
        if m["item"] is not None:
            final_results.append({
                "bbox": m["bbox"],
                "sku": m["item"]["sku"],
                "name": m["item"]["name"],
                "verified": m["item"]["verified"]
            })

    try:
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception: pass

    return {
        "status": "SUCCESS", 
        "total_items": len(final_results),
        "detections": final_results
    }


# --- TAREA: VIDEO TRACKING ---
@celery_app.task(name="tasks.process_video", bind=True)
def process_video_task(self, video_path):
    print(f"🎬 Iniciando procesamiento de video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "No se pudo abrir el video."}

    active_tracks = {}
    completed_tracks = {}
    
    max_unseen = vp_cfg['tracking']['max_unseen_frames']
    min_traj = vp_cfg['tracking']['min_trajectory_frames']
    margin_pct = vp_cfg['geometry']['edge_margin_pct']
    focus_top = vp_cfg['geometry']['focus_band_top_pct']
    focus_bot = vp_cfg['geometry']['focus_band_bottom_pct']
    min_h_pct = vp_cfg['geometry']['min_height_pct']
    
    dbscan_eps = vp_cfg.get('clustering', {}).get('dbscan_eps', 15.0)
    min_samples = vp_cfg.get('clustering', {}).get('min_samples', 2)
    deep_sim_threshold = vp_cfg.get('clustering', {}).get('deep_similarity_threshold', 0.75)

    frame_idx = 0
    trajectories = {} 

    # --- 1. TRACKING Y MEMORIA ESPACIO-TEMPORAL ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_idx += 1
            
        h_frame, w_frame = frame.shape[:2]
        margin_x, margin_y = int(w_frame * margin_pct), int(h_frame * margin_pct)
        f_top, f_bot = int(h_frame * focus_top), int(h_frame * focus_bot)
        min_bbox_h = int(h_frame * min_h_pct)

        results = yolo_model.track(frame, tracker=tracker_path, persist=True, verbose=False, conf=confidence_threshold, iou=iou_threshold, imgsz=imgsz, device='cuda')
        current_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None and results[0].masks is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            masks_xy = results[0].masks.xy 
            
            for box, tid, mask_pts in zip(boxes, track_ids, masks_xy):
                current_ids.add(tid)
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                
                box_h = y2 - y1
                cy = y1 + (box_h / 2.0)
                
                if (x2 - x1) < 30 or box_h < 30: continue
                
                if tid not in trajectories:
                    trajectories[tid] = {}
                trajectories[tid][frame_idx] = [x1, y1, x2, y2]
                
                is_near_edge = (x1 <= margin_x) or (y1 <= margin_y) or (x2 >= w_frame - margin_x) or (y2 >= h_frame - margin_y)
                in_focus = f_top < cy < f_bot
                is_foreground = box_h >= min_bbox_h
                
                if tid not in active_tracks:
                    active_tracks[tid] = {"score": -1, "crop": None, "unseen": 0, "seen": 1, "target": False, "best_frame": -1}
                else:
                    active_tracks[tid]["unseen"] = 0 
                    active_tracks[tid]["seen"] += 1
                
                if in_focus and is_foreground and not is_near_edge:
                    active_tracks[tid]["target"] = True
                
                if active_tracks[tid]["target"] and in_focus and not is_near_edge:
                    if len(mask_pts) > 0:
                        obj_mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                        pts = np.array(mask_pts, np.int32)
                        cv2.fillPoly(obj_mask, [pts], 255)
                        masked_frame = cv2.bitwise_and(frame, frame, mask=obj_mask)
                        crop = masked_frame[y1:y2, x1:x2]
                    else:
                        crop = frame[y1:y2, x1:x2]

                    sharpness = calculate_sharpness(crop)
                    if sharpness > active_tracks[tid]["score"]:
                        active_tracks[tid]["score"] = sharpness
                        active_tracks[tid]["crop"] = crop.copy() 
                        active_tracks[tid]["best_frame"] = frame_idx

        lost_tracks = []
        for tid in list(active_tracks.keys()):
            if tid not in current_ids:
                active_tracks[tid]["unseen"] += 1
                
            if active_tracks[tid]["unseen"] > max_unseen:
                t_data = active_tracks[tid]
                if t_data["crop"] is not None and t_data["seen"] >= min_traj and t_data["target"]: 
                    completed_tracks[tid] = {
                        "crop": t_data["crop"],
                        "best_frame": t_data["best_frame"]
                    }
                lost_tracks.append(tid)
                
        for tid in lost_tracks: del active_tracks[tid]

    for tid, t_data in active_tracks.items():
        if t_data["crop"] is not None and t_data["seen"] >= min_traj and t_data["target"]:
            completed_tracks[tid] = {
                "crop": t_data["crop"],
                "best_frame": t_data["best_frame"]
            }
        
    cap.release()
    
    # --- 2. PRIMARY RE-ID (Backend AI) ---
    self.update_state(state='PROCESSING', meta={'message': f'Extrayendo IDs: {len(completed_tracks)} productos encontrados.'})
    
    item_results = []
    task_map = {}
    reid_tasks = []
    
    try:
        for tid, t_data in completed_tracks.items():
            crop_rgb = cv2.cvtColor(t_data["crop"], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=95)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            payload = {"image_b64": img_b64, "bboxes": [[0, 0, pil_img.size[0], pil_img.size[1]]]}
            try:
                res = requests.post(f"{API_URL}/api/v1/predict", json=payload)
                task_id = res.json().get('task_id')
                reid_tasks.append(task_id)
                task_map[task_id] = {
                    "tid": tid,
                    "best_frame": t_data["best_frame"]
                }
            except Exception as e:
                print(f"Error encolando recorte a API: {e}")

        for t_id in reid_tasks:
            while True:
                res = requests.get(f"{API_URL}/api/v1/results/{t_id}")
                if res.status_code == 200:
                    data = res.json()
                    if data["status"] == "SUCCESS":
                        detections = data["result"].get("detections", [])
                        if detections:
                            pred = detections[0].get("prediction", {})
                            info = task_map[t_id]
                            info["sku"] = pred.get("sku", "Sin SKU")
                            info["name"] = pred.get("name", "Desconocido")
                            info["verified"] = pred.get("verified", False)
                            info["was_originally_verified"] = info["verified"]
                            info["embedding"] = pred.get("embedding", None)
                            item_results.append(info)
                        break
                    elif data["status"] == "FAILED":
                        print(f"Error en backend Re-ID: {data.get('error')}")
                        break
                time.sleep(0.5)

        # --- 3. CLUSTER FALLBACK (Motor Híbrido: DBSCAN + Semántica Dinámica) ---
        print("🧠 Interpolando trayectorias (BBoxes completos) para estabilizar tracks...")
        full_trajectories = {tid: interpolate_trajectory(traj) for tid, traj in trajectories.items()}
        
        unverified_items = [i for i in item_results if not i["verified"]]
        frames_to_process = set(u_item["best_frame"] for u_item in unverified_items)
        
        if frames_to_process:
            print(f"🔍 Evaluando {len(unverified_items)} productos dudosos usando Arquitectura Híbrida...")
        
        for f_target in frames_to_process:
            active_in_frame = []
            for item in item_results:
                tid = item["tid"]
                if f_target in full_trajectories[tid]:
                    bbox = full_trajectories[tid][f_target]
                    active_in_frame.append({
                        "item": item, 
                        "bbox": bbox
                    })
                    
            n_items = len(active_in_frame)
            if n_items < 2: continue 
                
            dist_matrix = np.zeros((n_items, n_items))
            for i in range(n_items):
                for j in range(i+1, n_items):
                    dist = bbox_distance(active_in_frame[i]["bbox"], active_in_frame[j]["bbox"])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
            
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric='precomputed')
            labels = clusterer.fit_predict(dist_matrix)
            
            for cluster_id in set(labels):
                if cluster_id == -1: continue 
                    
                cluster_members = [active_in_frame[i] for i in range(n_items) if labels[i] == cluster_id]
                anchors = [m for m in cluster_members if m["item"]["was_originally_verified"] and "Contexto" not in m["item"].get("name", "")]
                
                if anchors:
                    for member in cluster_members:
                        item_dict = member["item"]
                        
                        if not item_dict["verified"] and item_dict["best_frame"] == f_target:
                            
                            sorted_anchors = sorted(anchors, key=lambda a: (
                                bbox_distance(member["bbox"], a["bbox"]),
                                math.hypot( (member["bbox"][0]+member["bbox"][2])/2 - (a["bbox"][0]+a["bbox"][2])/2, 
                                             (member["bbox"][1]+member["bbox"][3])/2 - (a["bbox"][1]+a["bbox"][3])/2)
                            ))
                            
                            tid_dudoso = item_dict["tid"]
                            emb_dudoso = item_dict.get("embedding")
                            
                            # Obtenemos el crop para calcular el área visible
                            crop_dudoso = completed_tracks[tid_dudoso]["crop"]
                            vis_ratio = calculate_visible_ratio(crop_dudoso)
                            
                            rescued = False
                            
                            for a in sorted_anchors:
                                emb_ancla = a["item"].get("embedding")
                                sim_score = cosine_similarity(emb_dudoso, emb_ancla)
                                
                                # 🚀 LÓGICA DE UMBRAL DINÁMICO ESPACIAL
                                is_behind = is_queued_behind(a["bbox"], member["bbox"])
                                
                                if is_behind and vis_ratio < 0.40:
                                    dyn_thresh = 0.55
                                elif is_behind:
                                    dyn_thresh = 0.65
                                elif vis_ratio < 0.40:
                                    dyn_thresh = 0.70
                                else:
                                    dyn_thresh = deep_sim_threshold
                                
                                if sim_score >= dyn_thresh:
                                    item_dict["verified"] = True
                                    item_dict["sku"] = a["item"]["sku"]
                                    item_dict["name"] = f"{a['item']['name']} (Contexto Híbrido)"
                                    rescued = True
                                    break 

        # --- 4. AGREGACIÓN FINAL ---
        stock_count = {}
        for item in item_results:
            sku = item["sku"]
            name = item["name"]
            verified = item["verified"]
            
            agg_key = f"{sku}_{verified}"
            
            if agg_key not in stock_count:
                stock_count[agg_key] = {
                    "SKU": sku,
                    "Producto": name,
                    "Verificado": "Sí" if verified else "No (Requiere revisión)",
                    "Cantidad": 0
                }
            stock_count[agg_key]["Cantidad"] += 1

    finally:
        try: 
            if os.path.exists(video_path): os.remove(video_path)
        except Exception as e: pass
    
    return {
        "status": "SUCCESS",
        "total_unique_items_detected": len(completed_tracks),
        "stock_inventory": list(stock_count.values()) 
    }