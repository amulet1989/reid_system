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
from ultralytics import YOLO
print(f"🧠 Cargando modelo YOLO: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH)
tracker_path = f"/app/{vp_cfg['tracker_config']}"

# --- 2. CONFIGURACIÓN DE CELERY ---
celery_app = Celery(
    'video_worker',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
)

API_URL = os.getenv("API_URL", "http://api:8000")

def calculate_sharpness(img_crop):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

@celery_app.task(name="tasks.process_video", bind=True)
def process_video_task(self, video_path):
    print(f"🎬 Iniciando procesamiento de video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "No se pudo abrir el video."}

    active_tracks = {}
    completed_tracks = {}
    
    # Parámetros desde config
    max_unseen = vp_cfg['tracking']['max_unseen_frames']
    min_traj = vp_cfg['tracking']['min_trajectory_frames']
    margin_pct = vp_cfg['geometry']['edge_margin_pct']
    focus_top = vp_cfg['geometry']['focus_band_top_pct']
    focus_bot = vp_cfg['geometry']['focus_band_bottom_pct']
    min_h_pct = vp_cfg['geometry']['min_height_pct']

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        h_frame, w_frame = frame.shape[:2]
        margin_x, margin_y = int(w_frame * margin_pct), int(h_frame * margin_pct)
        f_top, f_bot = int(h_frame * focus_top), int(h_frame * focus_bot)
        min_bbox_h = int(h_frame * min_h_pct)
        
        results = yolo_model.track(frame, tracker=tracker_path, persist=True, verbose=False, conf=0.1, iou=0.7, imgsz=1024)
        current_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, tid in zip(boxes, track_ids):
                current_ids.add(tid)
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                
                box_h = y2 - y1
                cy = y1 + (box_h / 2.0)
                
                if (x2 - x1) < 30 or box_h < 30: continue
                
                is_near_edge = (x1 <= margin_x) or (y1 <= margin_y) or (x2 >= w_frame - margin_x) or (y2 >= h_frame - margin_y)
                in_focus = f_top < cy < f_bot
                is_foreground = box_h >= min_bbox_h
                
                if tid not in active_tracks:
                    active_tracks[tid] = {"score": -1, "crop": None, "unseen": 0, "seen": 1, "target": False}
                else:
                    active_tracks[tid]["unseen"] = 0 
                    active_tracks[tid]["seen"] += 1
                
                if in_focus and is_foreground and not is_near_edge:
                    active_tracks[tid]["target"] = True
                
                if active_tracks[tid]["target"] and in_focus and not is_near_edge:
                    crop = frame[y1:y2, x1:x2]
                    sharpness = calculate_sharpness(crop)
                    if sharpness > active_tracks[tid]["score"]:
                        active_tracks[tid]["score"] = sharpness
                        active_tracks[tid]["crop"] = crop.copy()

        lost_tracks = []
        for tid in list(active_tracks.keys()):
            if tid not in current_ids:
                active_tracks[tid]["unseen"] += 1
                
            if active_tracks[tid]["unseen"] > max_unseen:
                t_data = active_tracks[tid]
                if t_data["crop"] is not None and t_data["seen"] >= min_traj and t_data["target"]: 
                    completed_tracks[tid] = t_data["crop"]
                lost_tracks.append(tid)
                
        for tid in lost_tracks: del active_tracks[tid]

    for tid, t_data in active_tracks.items():
        if t_data["crop"] is not None and t_data["seen"] >= min_traj and t_data["target"]:
            completed_tracks[tid] = t_data["crop"]
        
    cap.release()
    
    # 3. LOTE DE PETICIONES AL API DE RE-ID
    self.update_state(state='PROCESSING', meta={'message': f'Extrayendo IDs: {len(completed_tracks)} productos encontrados.'})
    
    stock_count = {}
    reid_tasks = []
    
    try:
        # Enviar todos los crops a la cola general
        for tid, crop_bgr in completed_tracks.items():
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=95)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            payload = {"image_b64": img_b64, "bboxes": [[0, 0, pil_img.size[0], pil_img.size[1]]]}
            try:
                res = requests.post(f"{API_URL}/api/v1/predict", json=payload)
                reid_tasks.append(res.json().get('task_id'))
            except Exception as e:
                print(f"Error encolando recorte a API: {e}")

        # Polling para esperar que reid_backend termine
        for t_id in reid_tasks:
            while True:
                res = requests.get(f"{API_URL}/api/v1/results/{t_id}")
                if res.status_code == 200:
                    data = res.json()
                    if data["status"] == "SUCCESS":
                        detections = data["result"].get("detections", [])
                        if detections:
                            pred = detections[0].get("prediction", {})
                            
                            # Extraemos toda la data fundamental
                            sku = pred.get("sku", "Sin SKU")
                            name = pred.get("name", "Desconocido")
                            verified = pred.get("verified", False)
                            
                            # Creamos una llave única combinando SKU y Verificación
                            # Así separamos los "Takis verificados" de los "Takis dudosos"
                            agg_key = f"{sku}_{verified}"
                            
                            if agg_key not in stock_count:
                                stock_count[agg_key] = {
                                    "SKU": sku,
                                    "Producto": name,
                                    "Verificado": "Sí" if verified else "No (Requiere revisión)",
                                    "Cantidad": 0
                                }
                                
                            stock_count[agg_key]["Cantidad"] += 1
                        break
                    elif data["status"] == "FAILED":
                        print(f"Error en backend Re-ID: {data.get('error')}")
                        break
                time.sleep(0.5)

    finally:
        try: 
            if os.path.exists(video_path): os.remove(video_path)
        except Exception as e: pass
    
    return {
        "status": "SUCCESS",
        "total_unique_items_detected": len(completed_tracks),
        # Convertimos el diccionario en una lista plana para que la API la devuelva limpia
        "stock_inventory": list(stock_count.values()) 
    }