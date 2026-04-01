import os
import uuid
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import gc
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from lightglue import LightGlue, ALIKED, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd
import kornia.feature as KF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import models

# Importamos nuestra configuración central
from config_loader import cfg

# --- 1. VARIABLES GLOBALES Y CONFIGURACIÓN ---
DEVICE = cfg['system']['device']
CACHE_DIR = cfg['system']['cache_dir']
COLLECTION_NAME = cfg['database']['collection_name']

# Asegurar que el directorio de caché exista
os.makedirs(CACHE_DIR, exist_ok=True)

# Variables globales para los modelos y DB (se inicializan al arrancar el Worker)
qdrant = None
dinov2 = None
dinov2_transform = None
qwen_processor = None
qwen_model = None
extractor = None
matcher = None

# Clase para mantener el Aspect Ratio 
class SquarePad:
    def __call__(self, image):
        # image shape es [C, H, W] después de ToTensor()
        _, h, w = image.shape
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_right = max_wh - w - pad_left
        pad_top = (max_wh - h) // 2
        pad_bottom = max_wh - h - pad_top
        return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0)

# --- 2. INICIALIZACIÓN DEL SISTEMA ---
def init_system():
    """
    Carga los modelos en la GPU y luego conecta a Qdrant.
    Se ejecuta una sola vez cuando el Celery Worker arranca.
    """
    global qdrant, dinov2, dinov2_transform, qwen_processor, qwen_model, extractor, matcher

    print("🚀 Iniciando Sistema de Re-ID Multimodal...")

    # --- 1. PRIMERO CARGAMOS LOS MODELOS ---
    print("🧠 Cargando DINOv2...")
    dinov2 = torch.hub.load(cfg['models']['dinov2']['repo'], cfg['models']['dinov2']['name']).to(DEVICE)
    dinov2.eval()
    dino_size = cfg['models']['dinov2']['image_size']
    # Obtener de cfg si uso SquarePad o Resize
    use_square_pad = cfg['models']['dinov2'].get('use_square_pad', False)
    if use_square_pad:
        print("⚠️ DINOv2 configurado para usar SquarePad (manteniendo aspect ratio).")
        dinov2_transform = T.Compose([
            T.ToTensor(),
            SquarePad(), # Mantiene el aspect ratio original 
            T.Resize((dino_size, dino_size), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        print("⚠️ DINOv2 configurado para usar Resize directo (puede distorsionar imágenes).")
        dinov2_transform = T.Compose([
            T.ToTensor(),
            T.Resize((dino_size, dino_size), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    print("🧠 Cargando Qwen-VL...")
    qwen_id = cfg['models']['qwen']['id']
    qwen_processor = AutoProcessor.from_pretrained(qwen_id, trust_remote_code=True)
    qwen_model = AutoModel.from_pretrained(qwen_id, trust_remote_code=True).to(DEVICE)
    qwen_model.eval()

    # 4. Cargar Extractor Local y LightGlue -------------------------------------
    feat_type = cfg['models']['local_features']['type'].lower()
    print(f"🧠 Cargando Extractor Local ({feat_type.upper()}) y LightGlue...")
    
    max_kp = cfg['models']['local_features']['max_num_keypoints']
    det_thresh = cfg['models']['local_features']['detection_threshold']

    if feat_type == "dedode":
        extractor = KF.DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights="B-upright").eval().to(DEVICE)
        matcher = KF.LightGlue("dedodeb").eval().to(DEVICE) # ✅ La versión nativa (Drop-in replacement)
    elif feat_type == "aliked":
        extractor = ALIKED(max_num_keypoints=max_kp, detection_threshold=det_thresh).eval().to(DEVICE)
        matcher = LightGlue(features=feat_type).eval().to(DEVICE)
    elif feat_type == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_kp, nms_radius=4, keypoint_threshold=det_thresh).eval().to(DEVICE)
        matcher = LightGlue(features=feat_type).eval().to(DEVICE)
    elif feat_type in ["disk", "sift"]:
        # Otros modelos soportados
        matcher = LightGlue(features=feat_type).eval().to(DEVICE)
    else:
        raise ValueError(f"Extractor no soportado: {feat_type}")
    # --------------------------------------------------------------------

    print("✅ Todos los modelos cargados en GPU exitosamente.")

    # --- 2. LUEGO CONECTAMOS A DB E INGESTAMOS ---
    print("🔌 Conectando a Qdrant...")
    qhost = os.getenv("QDRANT_HOST", "vector_db")
    qport = int(os.getenv("QDRANT_PORT", 6333))
    qdrant = QdrantClient(host=qhost, port=qport)
    
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"📦 Creando colección '{COLLECTION_NAME}' en Qdrant...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dinov2": VectorParams(size=cfg['models']['dinov2']['vector_size'], distance=Distance.COSINE),
                "qwen_layout": VectorParams(size=cfg['models']['qwen']['vector_size'], distance=Distance.COSINE),
                "color_hsv": VectorParams(size=128, distance=Distance.COSINE) # 🚀 NUEVO
            }
        )
        print("📥 Colección nueva detectada. Ejecutando ingesta inicial del catálogo...")
        batch_ingest_catalog()
    else:
        print(f"✅ Colección '{COLLECTION_NAME}' detectada.")

# --- 3. FUNCIONES DE EXTRACCIÓN (EN MEMORIA Y DISCO) ---
def get_dinov2_embedding_from_array(img_rgb):
    img_tensor = dinov2_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        emb = dinov2(img_tensor)
    return (emb / emb.norm(dim=-1, keepdim=True))[0].cpu().numpy()

def get_qwen_layout_embedding_from_array(img_rgb, metadata_text=""):
    image = Image.fromarray(img_rgb)
    
    # --- PARCHE DE ROTACIÓN EXIF ---
    image = ImageOps.exif_transpose(image) # ¡VITAL! Corregir orientación cruda

    max_size = cfg['models']['qwen']['max_image_size']
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    # raw_prompt = f"Extract all text, brand names, and analyze the layout of this retail product packaging. Context: {metadata_text}"
    # 🚀 NUEVO: Leemos el prompt desde YAML y le inyectamos el contexto
    base_prompt = cfg['models']['qwen']['prompt']
    raw_prompt = base_prompt.format(metadata_text=metadata_text)
    # print(f"📝 Prompt para Qwen-VL: {raw_prompt}")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": raw_prompt}]}]
    
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Subimos el inference_mode para envolver todo el paso por la GPU
    with torch.inference_mode():
        inputs = qwen_processor(images=image, text=text_prompt, return_tensors="pt", padding=True).to(DEVICE)
        outputs = qwen_model(**inputs)
        emb = outputs.last_hidden_state[:, -1, :] 
    
    emb = emb.float()
    return (emb / emb.norm(dim=-1, keepdim=True))[0].cpu().numpy()

@torch.inference_mode()
def extract_local_features_from_array(img_rgb):
    img = img_rgb.copy()
    max_dim = cfg['models']['local_features']['max_image_size']

    # --- [NUEVO] PREPROCESAMIENTO DINÁMICO ---
    if cfg['models']['local_features']['preprocess'].get('use_clahe', False):
        # print("⚙️ Aplicando preprocesamiento CLAHE para mejorar la detección de keypoints...")
        # SuperPoint prefiere intensidad de textura sobre color
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(gray)
        # Convertimos de vuelta a RGB para mantener compatibilidad con el resto del pipeline
        img = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    # -----------------------------------------

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    feat_type = cfg['models']['local_features']['type'].lower()
    
    if feat_type == "dedode":
        image_tensor = T.ToTensor()(img).unsqueeze(0).to(DEVICE)  # [0,1] → DeDoDe lo normaliza internamente
    
        max_kp = cfg['models']['local_features']['max_num_keypoints']
    
        # with torch.inference_mode():
        keypoints, scores = extractor.detect(image_tensor, n=max_kp)
        descriptors = extractor.describe(image_tensor, keypoints=keypoints)
    
        # 1. Aseguramos batch dim (DeDoDe devuelve sin batch aunque la entrada tenga B=1)
        if keypoints.ndim == 2:
            keypoints = keypoints.unsqueeze(0)
            scores = scores.unsqueeze(0)
            descriptors = descriptors.unsqueeze(0)
    
        # 2. ¡LA CLAVE! Convertimos coordenadas normalizadas [0,1] → píxeles absolutos
        #    (igual que ALIKED/SuperPoint, para que homografía y MAGSAC funcionen)
        image_size_tensor = torch.tensor([img.shape[1], img.shape[0]], device=DEVICE).view(1, 2)
        keypoints = keypoints * image_size_tensor.unsqueeze(1)   # broadcasting mágico
    
        feats = {
            "keypoints": keypoints,      # ← ahora en píxeles
            "descriptors": descriptors,
            "scores": scores,
            "image_size": image_size_tensor
        }
    else:
        # 2. Librería LightGlue estándar (ALIKED, SuperPoint, DISK, SIFT)
        # Esta librería prefiere su propio conversor
        image_tensor = numpy_image_to_torch(img).to(DEVICE)
        feats = extractor.extract(image_tensor)
        
    return feats

# Embeddings de color
def get_hsv_color_embedding(img_rgb, is_query=False):
    """
    Calcula un Histograma HSV 3D invariante a la escala y blindado contra NaN.
    """
    img_work = img_rgb.copy()
    
    # 1. Recorte Central Geométrico (Solo para queries)
    if is_query:
        h, w = img_work.shape[:2]
        m_y, m_x = int(h * 0.15), int(w * 0.15)
        
        # Verificamos que al recortar quede una imagen de al menos 10x10 px
        if (h - 2*m_y) > 10 and (w - 2*m_x) > 10: 
            img_work = img_work[m_y:h-m_y, m_x:w-m_x]
    else:        # Para las imágenes de referencia, aplicamos un recorte más agresivo para enfocarnos en el centro del empaque
        h, w = img_work.shape[:2]
        m_y, m_x = int(h * 0.25), int(w * 0.25)
        
        if (h - 2*m_y) > 10 and (w - 2*m_x) > 10:
            img_work = img_work[m_y:h-m_y, m_x:w-m_x]

    img_hsv = cv2.cvtColor(img_work, cv2.COLOR_RGB2HSV)
    
    # 2. Máscara de Color Puro (Ignorar blancos, negros, grises y destellos)
    # H: 0-180 (Todos los colores)
    # S: 30-255 (Ignorar grises y el fondo blanco del catálogo)
    # V: 30-240 (Ignorar sombras profundas y destellos del plástico)
    lower_bound = np.array([0, 10, 15])
    upper_bound = np.array([180, 255, 250])
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    # 3. Histograma 3D
    hist = cv2.calcHist(
        [img_hsv], 
        channels=[0, 1, 2], 
        mask=mask, 
        histSize=[8, 4, 4], 
        ranges=[0, 180, 0, 256, 0, 256]
    )
    
    # 4. 🚀 DEFENSA CONTRA EL VACÍO (Pixel Starvation)
    # Si la suma del histograma es 0, no hay colores útiles. 
    # Devolvemos un vector neutro para evitar errores NaN en el Coseno.
    if hist.sum() == 0:
        return np.zeros(128, dtype=np.float32)
    
    # 5. Normalización L2 (Invarianza de Escala)
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
    
    return hist.flatten()

# --- 4. FUNCIONES DE INGESTA (OFFLINE) ---
def index_product_offline(sku, name, category, image_path, view_name="frente"):
    
    try:
        pil_img = Image.open(image_path).convert("RGB")
        # --- PARCHE DE ROTACIÓN EXIF ---
        pil_img = ImageOps.exif_transpose(pil_img) # Hornear rotación física
        # -------------------------------
        img_rgb = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"No se pudo leer la imagen con Pillow: {image_path}. Error: {e}")

    try:
        vec_visual = get_dinov2_embedding_from_array(img_rgb)
        # context = "Reference retail product packaging from official catalog."
        # context = "Empaque de producto minorista de referencia del catálogo oficial."
        context = cfg['models']['qwen']['context']['reference']
        vec_layout = get_qwen_layout_embedding_from_array(img_rgb, metadata_text=context)
        
        feats = extract_local_features_from_array(img_rgb)
        feats_cpu = {k: v.cpu() for k, v in feats.items() if isinstance(v, torch.Tensor)}
        
        feat_type = cfg['models']['local_features']['type'].lower()
        safe_img_name = os.path.basename(image_path).replace(".", "_")
        feat_path = os.path.join(CACHE_DIR, f"{sku}_{safe_img_name}_{feat_type}.pt") # Nombre dinámico
        torch.save(feats_cpu, feat_path)
        
        vec_color = get_hsv_color_embedding(img_rgb) # 🚀 NUEVO: Embedding de color HSV

        point_id = str(uuid.uuid4())
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=point_id, 
                vector={
                    "dinov2": vec_visual.tolist(), 
                    "qwen_layout": vec_layout.tolist(), 
                    "color_hsv": vec_color.tolist()}, 
                payload={
                    "sku": sku, 
                    "name": name, 
                    "view": view_name,
                    "image_path": image_path,
                    "feature_path": feat_path
                }
            )]
        )
    finally:
        # 🧹 LIMPIEZA EXTREMA POR CADA PRODUCTO INGESTADO
        if 'feats' in locals(): del feats
        if 'feats_cpu' in locals(): del feats_cpu
        if 'img_rgb' in locals(): del img_rgb
        
        gc.collect() # Obliga a Python a limpiar la RAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Obliga a la GPU a limpiar la VRAM

def batch_ingest_catalog():
    catalog_dir = cfg['system']['catalog_dir']
    csv_path = cfg['system']['csv_path']
    print(f"📂 Iniciando ingesta masiva desde: {catalog_dir}")
    
    try:
        df = pd.read_csv(csv_path, sep=',', dtype=str, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=',', dtype=str, encoding='latin1')
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {csv_path}")
        return

    df.columns = df.columns.str.strip()
    if 'SKU' not in df.columns:
        print("❌ Error: Columna 'SKU' no encontrada en el CSV.")
        return

    catalog_metadata = {}
    for _, row in df.iterrows():
        if pd.isna(row['SKU']): continue
        raw_sku = str(row['SKU']).strip()
        stripped_sku = raw_sku.lstrip('0') 
        name = str(row.get('DESCRIPCIÓN DE PRODUCTO', str(row.get('DESCRIPCION DE PRODUCTO', 'Sin nombre')))).strip()
        category = str(row.get('CATEGORÍA', str(row.get('CATEGORIA', 'Sin categoría')))).strip()
        catalog_metadata[stripped_sku] = {"name": name, "category": category}

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if not os.path.exists(catalog_dir):
        print(f"❌ Carpeta no encontrada: {catalog_dir}")
        return
        
    folders = [f for f in os.listdir(catalog_dir) if os.path.isdir(os.path.join(catalog_dir, f))]

    for folder_name in tqdm(folders, desc="Ingestando"):
        stripped_folder_sku = folder_name.lstrip('0')
        if stripped_folder_sku not in catalog_metadata: continue

        sku_path = os.path.join(catalog_dir, folder_name)
        images = [img for img in os.listdir(sku_path) if os.path.splitext(img)[1].lower() in valid_extensions]
        
        meta = catalog_metadata[stripped_folder_sku]
        for img_name in images:
            img_path = os.path.join(sku_path, img_name)
            view_name = os.path.splitext(img_name)[0] 
            try:
                index_product_offline(folder_name, meta["name"], meta["category"], img_path, view_name)
            except Exception as e:
                print(f"❌ Error en {img_name} ({folder_name}): {e}")

# --- 5. FUNCIONES DE CONSULTA (ONLINE) ---
def crop_bbox(image_rgb, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image_rgb[y1:y2, x1:x2]

# Reranking local k-reciprocal basado en DINOv2 + Qwen-VL, pero solo entre los candidatos top-K recuperados por Qdrant.
def local_k_reciprocal_re_ranking(vec_visual, vec_layout, search_results):
    """
    Aplica el algoritmo k-reciprocal re-ranking (Zhong et al.) de forma ultra-ligera
    restringido al sub-espacio topológico de los candidatos recuperados por Qdrant.
    """
    K = len(search_results)
    if K == 0:
        return search_results
        
    cfg_rr = cfg['pipeline']['retrieval']['re_ranking']
    k1 = min(cfg_rr['k1'], K)
    lambda_weight = cfg_rr['lambda_weight']
    
    dim_dino = len(vec_visual)
    dim_qwen = len(vec_layout)
    
    # 1. Construir matrices locales: [Query, Cand_1, Cand_2, ..., Cand_K]
    all_dino = np.zeros((K + 1, dim_dino))
    all_qwen = np.zeros((K + 1, dim_qwen))
    
    all_dino[0] = vec_visual
    all_qwen[0] = vec_layout
    
    for i, hit in enumerate(search_results):
        vecs = hit.vector
        all_dino[i+1] = vecs["dinov2"]
        all_qwen[i+1] = vecs["qwen_layout"]
        
    # Normalización L2 segura
    all_dino = all_dino / np.linalg.norm(all_dino, axis=1, keepdims=True)
    all_qwen = all_qwen / np.linalg.norm(all_qwen, axis=1, keepdims=True)
    
    # 2. Calcular Distancia Original (Promedio de Cosine Distances multimodales)
    sim_dino = np.dot(all_dino, all_dino.T)
    sim_qwen = np.dot(all_qwen, all_qwen.T)
    dist_matrix = 1.0 - ((sim_dino + sim_qwen) / 2.0) # Distancia original d(p, g_i)
    
    # 3. Encontrar vecindarios recíprocos y aplicar pesos Gaussianos
    initial_rank = np.argsort(dist_matrix, axis=1) 
    V = np.zeros((K + 1, K + 1))
    
    for i in range(K + 1):
        forward_k_neighbors = initial_rank[i, :k1 + 1] 
        for j in forward_k_neighbors:
            backward_k_neighbors = initial_rank[j, :k1 + 1]
            if i in backward_k_neighbors:
                # Es recíproco. Aplicar Eq. 7 del paper
                V[i, j] = np.exp(-dist_matrix[i, j])
                
    # 4. Calcular Distancia de Jaccard y Distancia Final (Eq. 10 y Eq. 12)
    V_probe = V[0] 
    re_ranked_hits = []
    
    for i in range(1, K + 1):
        V_candidate = V[i]
        
        intersection = np.sum(np.minimum(V_probe, V_candidate))
        union = np.sum(np.maximum(V_probe, V_candidate))
        
        jaccard_dist = 1.0 if union == 0 else 1.0 - (intersection / union)
        
        # Eq. 12 del paper: Combinación ponderada
        final_dist = (1 - lambda_weight) * jaccard_dist + lambda_weight * dist_matrix[0, i]
        
        # Convertir distancia a Score (mayor es mejor) para la lógica downstream
        final_score = 1.0 - final_dist
        
        # Actualizamos el score del candidato
        hit = search_results[i-1]
        hit.score = float(final_score)
        re_ranked_hits.append(hit)
        
    # Ordenar por el nuevo score k-recíproco
    re_ranked_hits.sort(key=lambda x: x.score, reverse=True)
    return re_ranked_hits

def query_online_multimodal_cached(img_crop_rgb):
    vec_visual = get_dinov2_embedding_from_array(img_crop_rgb)
    context = cfg['models']['qwen']['context']['query']
    vec_layout = get_qwen_layout_embedding_from_array(img_crop_rgb, metadata_text=context)
    vec_query_color = get_hsv_color_embedding(img_crop_rgb, is_query=True) # 🚀 NUEVO: Embedding de color HSV
    # Consulta de empaque de producto minorista desde el estante de la tienda.
    # Query retail product packaging from store shelf.
    
    top_k = cfg['pipeline']['retrieval']['top_k']
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=vec_visual.tolist(), using="dinov2", limit=top_k),
            models.Prefetch(query=vec_layout.tolist(), using="qwen_layout", limit=top_k)
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_vectors=True 
    ).points
    
    if not search_results:
        return None
    
    # 🚀 NUEVO: Bloque de Re-Ranking K-Recíproco Topológico
    if cfg['pipeline']['retrieval'].get('re_ranking', {}).get('enabled', False):
        search_results = local_k_reciprocal_re_ranking(vec_visual, vec_layout, search_results)
        
    top_embedding_hit = search_results[0]
    query_feats = extract_local_features_from_array(img_crop_rgb)
    
    best_match = None
    min_inliers_valid = cfg['pipeline']['verification']['min_inliers_valid']
    max_inliers_found = 0 

    feat_type = cfg['models']['local_features']['type'].lower()

    
    # Lista para guardar todos los candidatos que superen la prueba geométrica
    valid_matches = []
    
    # Verificación Fina (¡Unificada para todos los modelos!)
    for hit in search_results:
        cand_feat_path = hit.payload.get("feature_path")
        
        if not cand_feat_path:
            continue
            
        try:
            cand_feats = torch.load(cand_feat_path, map_location=DEVICE, weights_only=False)
        except Exception:
            continue

        with torch.inference_mode():
            matches01_raw = matcher({"image0": query_feats, "image1": cand_feats})
    
            if feat_type == "dedode":
                matches = matches01_raw["matches"][0]
            else:
                matches01 = rbd(matches01_raw)
                matches = matches01['matches'] 
    
            # 1. FILTRO PREVIO: Si LightGlue no ve nada, fuera.
            num_matches = len(matches)
            if num_matches < cfg['pipeline']['verification']['min_matches_lightglue']:
                continue
        
            pts0 = query_feats['keypoints'][0][matches[:, 0]].cpu().numpy()
            pts1 = cand_feats['keypoints'][0][matches[:, 1]].cpu().numpy()
        
        H, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, cfg['pipeline']['verification']['ransac_threshold'])
        if mask is None:
            continue
            
        inliers = mask.sum()
                
        # --- [PUNTO 1: CONFIDENCE RATIO] ---
        # Calculamos qué tan "limpio" es el match
        conf_ratio = inliers / num_matches if num_matches > 0 else 0
        
        # Definimos si este candidato es GEOMÉTRICAMENTE SÓLIDO
        # Regla: O tienes muchos inliers (>25) o tienes un ratio de éxito muy alto (>45%)
        is_geometric_valid = (inliers >= min_inliers_valid) or (inliers >= 10 and conf_ratio > 0.5)
        
        if inliers > max_inliers_found:
            max_inliers_found = inliers

        if is_geometric_valid:
            match_data = hit.payload.copy()
            match_data["color_hsv"] = hit.vector["color_hsv"] # Rescatamos el vector de color de Qdrant y lo guardamos en nuestro diccionario
            match_data["inliers"] = int(inliers)  # 🔥 El cast a int() de Python es vital aquí
            match_data["conf_ratio"] = float(conf_ratio) # 🔥 Cast a float nativo por seguridad
            match_data["fusion_score"] = float(hit.score)
            valid_matches.append(match_data)
                
    # 🧠 LÓGICA DE DESEMPATE MULTIMODAL (TIE-BREAKER)
    if valid_matches:
        # 1. Ordenamos por inliers de mayor a menor para ver quién sacó la nota más alta
        valid_matches.sort(key=lambda x: x["inliers"], reverse=True)
        top_inliers = valid_matches[0]["inliers"]
        
        # 2. Definimos un "margen de empate" (ej. diferencia de 15% respecto al mejor, con un mínimo de 5 inliers)
        # Si el mejor tiene 100 inliers, cualquiera con 85+ está en el empate.
        # Si el mejor tiene 18, cualquiera con 13+ (18-5) entra al desempate.
        margin = max(5, int(top_inliers * 0.15))
        # margin = max(4, int(top_inliers * 0.10))
        
        # 3. Filtramos los candidatos que están dentro de este margen de empate técnico
        tied_candidates = [m for m in valid_matches if m["inliers"] >= (top_inliers - margin)]
        
        # # 4. De los empatados geométricamente, DINOv2 y Qwen eligen al ganador por color/texto
        # best_match = max(tied_candidates, key=lambda x: x["fusion_score"])
        
        # 🚀 REGLA DEL COLOR: Desempate Fantasma (Sin afectar la verificación)
        for cand in tied_candidates:
            # Recuperamos el vector de Qdrant (Asegúrate de haberlo guardado antes en el dict)
            cand_color = np.array(cand.get("color_hsv", np.zeros(128)))
            
            # 1. DEFENSA CONTRA VECTORES VACÍOS
            if np.sum(vec_query_color) == 0 or np.sum(cand_color) == 0:
                cand["color_score"] = 0.0
            else:
                cand["color_score"] = float(np.dot(vec_query_color, cand_color))
            
            # 2. CREACIÓN DEL "SHADOW SCORE" PARA EL DESEMPATE
            # Partimos del score original, pero en una variable paralela
            cand["selection_score"] = cand["fusion_score"]
            
            # Si el color coincide moderadamente bien (luces similares), le damos un bonus para ganar
            if cand["color_score"] >= 0.30:
                cand["selection_score"] *= 1.2
                
            # Si el color es radicalmente opuesto (Gemelo Malvado evidente), lo hundimos
            elif cand["color_score"] < 0.10:
                cand["selection_score"] *= 0.5
                
        # 🧠 EL TIE-BREAKER DEFINITIVO
        # Elegimos al ganador usando nuestra variable fantasma alterada por el color...
        best_match = max(tied_candidates, key=lambda x: x["selection_score"])
   
        
        # 1. Extraemos los valores clave del ganador
        final_inliers = best_match["inliers"]
        final_score = best_match["fusion_score"]
        
        # 2. VETO DINÁMICO (La magia de confiar en la geometría)
        best_match["verified"] = False # Asumimos falso por defecto
        
        if final_inliers >= 20:
            # GEOMETRÍA ABRUMADORA
            # Si RANSAC encontró más de 20 puntos alineados, es casi imposible que sea un error.
            # Solo vetamos si el score multimodal es catastróficamente bajo (< 0.10)
            if final_score >= 0.10:
                best_match["verified"] = True
                
        elif final_inliers >= 14:
            # GEOMETRÍA DUDOSA ("La Zona de Peligro")
            # En tus datos, aquí están los errores entre productos hermanos.
            # Necesitamos que DINO/Qwen estén MUY seguros para aprobarlo.
            if final_score >= 0.35: 
                best_match["verified"] = True       
        else:
            # INLIERS BAJOS (< 14)
            # Nunca lo verificamos, es puro ruido estadístico.
            best_match["verified"] = False
            
        return best_match
        
    else:
        # Fallback si nadie superó los inliers mínimos (La geometría falló)
        # 1. Definimos un margen estrecho para el empate técnico (ej. 5% del mejor score)
        top_score = search_results[0].score
        margin = top_score * 0.05 
        
        # Filtramos los hits crudos de Qdrant que están en la zona de empate
        fallback_candidates = [hit for hit in search_results if hit.score >= (top_score - margin)]
        
        best_fallback_hit = fallback_candidates[0]
        best_shadow_score = -1.0
        final_color_score = 0.0
        
        # 2. Re-ranking por color en las sombras
        for hit in fallback_candidates:
            # En search_results, los vectores están en el atributo .vector
            cand_color = np.array(hit.vector.get("color_hsv", np.zeros(128)))
            
            if np.sum(vec_query_color) == 0 or np.sum(cand_color) == 0:
                color_score = 0.0
            else:
                color_score = float(np.dot(vec_query_color, cand_color))
                
            shadow_score = hit.score
            
            # Premiamos o castigamos igual que en la verificación
            if color_score >= 0.30:
                shadow_score *= 1.2
            elif color_score < 0.10:
                shadow_score *= 0.5
                
            # Coronamos al nuevo ganador del fallback
            if shadow_score > best_shadow_score:
                best_shadow_score = shadow_score
                best_fallback_hit = hit
                final_color_score = color_score

        # 3. Armamos el diccionario final de respuesta con el ganador del re-ranking
        fallback_match = best_fallback_hit.payload.copy()
        fallback_match["fusion_score"] = float(best_fallback_hit.score) # Mantenemos el score real intacto
        fallback_match["color_score"] = float(final_color_score)
        fallback_match["inliers"] = int(max_inliers_found)
        fallback_match["verified"] = False # Siempre falso porque falló la geometría
        
        return fallback_match
    
    