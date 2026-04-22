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
    
# --- CONTROLADORES DINÁMICOS DE VRAM ---
def load_model_dynamically(model_name):
    """Carga un modelo a VRAM solo si no existe."""
    global dinov2, dinov2_transform, qwen_processor, qwen_model
    
    if model_name == "dinov2" and dinov2 is None:
        print("⏳ Carga Efímera: Levantando DINOv2 a VRAM...")
        dinov2 = torch.hub.load(cfg['models']['dinov2']['repo'], cfg['models']['dinov2']['name']).to(DEVICE)
        dinov2.eval()
        dino_size = cfg['models']['dinov2']['image_size']
        use_square_pad = cfg['models']['dinov2'].get('use_square_pad', False)
        if use_square_pad:
            dinov2_transform = T.Compose([T.ToTensor(), SquarePad(), T.Resize((dino_size, dino_size), antialias=True), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        else:
            dinov2_transform = T.Compose([T.ToTensor(), T.Resize((dino_size, dino_size), antialias=True), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    elif model_name == "qwen" and qwen_model is None:
        print("⏳ Carga Efímera: Levantando Qwen-VL a VRAM...")
        qwen_id = cfg['models']['qwen']['id']
        qwen_processor = AutoProcessor.from_pretrained(qwen_id, trust_remote_code=True)
        qwen_model = AutoModel.from_pretrained(qwen_id, trust_remote_code=True).to(DEVICE)
        qwen_model.eval()

def unload_model_dynamically(model_name):
    """Destruye un modelo de la VRAM y fuerza la limpieza del sistema."""
    global dinov2, dinov2_transform, qwen_processor, qwen_model
    
    if model_name == "dinov2" and dinov2 is not None:
        print("🧹 Descarga Efímera: Eliminando DINOv2 de VRAM...")
        del dinov2
        dinov2 = None
        dinov2_transform = None
        
    elif model_name == "qwen" and qwen_model is not None:
        print("🧹 Descarga Efímera: Eliminando Qwen-VL de VRAM...")
        del qwen_model
        qwen_model = None
        qwen_processor = None
        
    # Obligamos a Python y a CUDA a liberar los punteros huérfanos
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --- 2. INICIALIZACIÓN DEL SISTEMA ---
def init_system():
    """
    Carga los modelos en la GPU y luego conecta a Qdrant.
    Se ejecuta una sola vez cuando el Celery Worker arranca.
    """
    global qdrant, extractor, matcher

    print("🚀 Iniciando Sistema de Re-ID Multimodal...")
    search_mode = cfg['pipeline']['retrieval'].get('search_mode', 'fusion').lower()

    # --- 1. CARGA CONDICIONAL DE MODELOS ---
    if search_mode in ["fusion", "dinov2_only"]:
        load_model_dynamically("dinov2")
    else:
        print("⏭️ AHORRO VRAM: Omitiendo DINOv2.")

    if search_mode in ["fusion", "qwen_only"]:
        load_model_dynamically("qwen")
    else:
        print("⏭️ AHORRO VRAM: Omitiendo Qwen-VL.")

    # Extractor Local y LightGlue (Siempre requeridos)
    feat_type = cfg['models']['local_features']['type'].lower()
    print(f"🧠 Cargando Extractor Local ({feat_type.upper()}) y LightGlue...")
    
    max_kp = cfg['models']['local_features']['max_num_keypoints']
    det_thresh = cfg['models']['local_features']['detection_threshold']

    if feat_type == "dedode":
        extractor = KF.DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights="B-upright").eval().to(DEVICE)
        matcher = KF.LightGlue("dedodeb").eval().to(DEVICE)
    elif feat_type == "aliked":
        extractor = ALIKED(max_num_keypoints=max_kp, detection_threshold=det_thresh).eval().to(DEVICE)
        matcher = LightGlue(features=feat_type).eval().to(DEVICE)
    elif feat_type == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_kp, nms_radius=4, keypoint_threshold=det_thresh).eval().to(DEVICE)
        matcher = LightGlue(features=feat_type).eval().to(DEVICE)
    elif feat_type in ["disk", "sift"]:
        matcher = LightGlue(features=feat_type).eval().to(DEVICE)
    else:
        raise ValueError(f"Extractor no soportado: {feat_type}")

    print("✅ Modelos base cargados en GPU exitosamente.")

    # --- 2. CONECTAMOS A DB E INGESTAMOS ---
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
                "color_hsv": VectorParams(size=128, distance=Distance.COSINE)
            }
        )
        print("📥 Colección nueva detectada. Ejecutando ingesta inicial del catálogo...")
        batch_ingest_catalog()
    else:
        print(f"✅ Colección '{COLLECTION_NAME}' detectada.")

# --- 3. FUNCIONES DE EXTRACCIÓN (EN MEMORIA Y DISCO) ---
def get_dinov2_embedding_from_array(img_rgb):
    global dinov2
    if dinov2 is None:
        # Generar un vector "dummy" normalizado para no romper Qdrant
        dim = cfg['models']['dinov2']['vector_size']
        return (np.ones(dim, dtype=np.float32) / np.sqrt(dim))
        
    img_tensor = dinov2_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        emb = dinov2(img_tensor)
    return (emb / emb.norm(dim=-1, keepdim=True))[0].cpu().numpy()

def get_qwen_layout_embedding_from_array(img_rgb, metadata_text=""):
    global qwen_model
    if qwen_model is None:
        dim = cfg['models']['qwen']['vector_size']
        return (np.ones(dim, dtype=np.float32) / np.sqrt(dim))
        
    image = Image.fromarray(img_rgb)
    image = ImageOps.exif_transpose(image)

    max_size = cfg['models']['qwen']['max_image_size']
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    base_prompt = cfg['models']['qwen']['prompt']
    raw_prompt = base_prompt.format(metadata_text=metadata_text)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": raw_prompt}]}]
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
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
    search_mode = cfg['pipeline']['retrieval'].get('search_mode', 'fusion').lower()
    
    print(f"📂 Iniciando ingesta masiva desde: {catalog_dir}")
    
    # 🚀 INFLAR VRAM: Aseguramos que TODOS los modelos estén en memoria para el catálogo
    load_model_dynamically("dinov2")
    load_model_dynamically("qwen")
    
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

    # 🚀 DESINFLAR VRAM: Restauramos el estado de memoria según el config.yml
    print("⚖️ Ingesta finalizada. Restaurando política de VRAM...")
    if search_mode == "qwen_only":
        unload_model_dynamically("dinov2")
    elif search_mode == "dinov2_only":
        unload_model_dynamically("qwen")

# --- 5. FUNCIONES DE CONSULTA (ONLINE) ---
def crop_bbox(image_rgb, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image_rgb[y1:y2, x1:x2]


# Reranking local k-reciprocal adaptado para soportar Single-Model o Fusion
def local_k_reciprocal_re_ranking(vec_visual, vec_layout, search_results):
    """
    Aplica el algoritmo k-reciprocal re-ranking.
    Soporta dinámicamente si falta alguno de los embeddings (modos single-model).
    """
    K = len(search_results)
    if K == 0:
        return search_results
        
    cfg_rr = cfg['pipeline']['retrieval']['re_ranking']
    k1 = min(cfg_rr['k1'], K)
    lambda_weight = cfg_rr['lambda_weight']
    
    sim_dino = 0
    sim_qwen = 0
    divisor = 0
    
    # 1. Matriz DINOv2 (Solo si existe)
    if vec_visual is not None:
        dim_dino = len(vec_visual)
        all_dino = np.zeros((K + 1, dim_dino))
        all_dino[0] = vec_visual
        for i, hit in enumerate(search_results):
            all_dino[i+1] = hit.vector["dinov2"]
        all_dino = all_dino / np.linalg.norm(all_dino, axis=1, keepdims=True)
        sim_dino = np.dot(all_dino, all_dino.T)
        divisor += 1

    # 2. Matriz Qwen (Solo si existe)
    if vec_layout is not None:
        dim_qwen = len(vec_layout)
        all_qwen = np.zeros((K + 1, dim_qwen))
        all_qwen[0] = vec_layout
        for i, hit in enumerate(search_results):
            all_qwen[i+1] = hit.vector["qwen_layout"]
        all_qwen = all_qwen / np.linalg.norm(all_qwen, axis=1, keepdims=True)
        sim_qwen = np.dot(all_qwen, all_qwen.T)
        divisor += 1
        
    if divisor == 0:
        return search_results
        
    # 3. Distancia Original (Promedio dinámico según lo que esté activo)
    dist_matrix = 1.0 - ((sim_dino + sim_qwen) / float(divisor)) 
    
    # 4. Encontrar vecindarios recíprocos y aplicar pesos Gaussianos
    initial_rank = np.argsort(dist_matrix, axis=1) 
    V = np.zeros((K + 1, K + 1))
    
    for i in range(K + 1):
        forward_k_neighbors = initial_rank[i, :k1 + 1] 
        for j in forward_k_neighbors:
            backward_k_neighbors = initial_rank[j, :k1 + 1]
            if i in backward_k_neighbors:
                V[i, j] = np.exp(-dist_matrix[i, j])
                
    # 5. Calcular Distancia de Jaccard y Distancia Final
    V_probe = V[0] 
    re_ranked_hits = []
    
    for i in range(1, K + 1):
        V_candidate = V[i]
        intersection = np.sum(np.minimum(V_probe, V_candidate))
        union = np.sum(np.maximum(V_probe, V_candidate))
        jaccard_dist = 1.0 if union == 0 else 1.0 - (intersection / union)
        final_dist = (1 - lambda_weight) * jaccard_dist + lambda_weight * dist_matrix[0, i]
        final_score = 1.0 - final_dist
        
        hit = search_results[i-1]
        hit.score = float(final_score)
        re_ranked_hits.append(hit)
        
    re_ranked_hits.sort(key=lambda x: x.score, reverse=True)
    return re_ranked_hits

    
def query_online_multimodal_cached(img_crop_rgb):
    # 🚀 NUEVO: Leemos el modo de búsqueda desde config
    search_mode = cfg['pipeline']['retrieval'].get('search_mode', 'fusion').lower()
    top_k = cfg['pipeline']['retrieval']['top_k']
    
    vec_visual = None
    vec_layout = None
    
    # 🚀 EXTRACCIÓN CONDICIONAL (Ahorro de GPU)
    if search_mode in ["fusion", "dinov2_only"]:
        vec_visual = get_dinov2_embedding_from_array(img_crop_rgb)
        
    if search_mode in ["fusion", "qwen_only"]:
        context = cfg['models']['qwen']['context']['query']
        vec_layout = get_qwen_layout_embedding_from_array(img_crop_rgb, metadata_text=context)
        
    vec_query_color = get_hsv_color_embedding(img_crop_rgb, is_query=True)
    
    # 🚀 BÚSQUEDA QDRANT CONDICIONAL
    if search_mode == "dinov2_only":
        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec_visual.tolist(),
            using="dinov2",
            limit=top_k,
            with_vectors=True 
        ).points
    elif search_mode == "qwen_only":
        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec_layout.tolist(),
            using="qwen_layout",
            limit=top_k,
            with_vectors=True 
        ).points
    else: # Modo Fusion por defecto
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
        
    # Elegimos qué embedding enviar de regreso al worker de video para el cortafuegos semántico
    # Preferimos Qwen 
    returned_embedding = vec_layout if vec_layout is not None else vec_visual

    if not search_results:
        return {
            "sku": "N/A",
            "name": "Desconocido",
            "verified": False,
            "embedding": returned_embedding.tolist() 
        }
    
    if cfg['pipeline']['retrieval'].get('re_ranking', {}).get('enabled', False):
        search_results = local_k_reciprocal_re_ranking(vec_visual, vec_layout, search_results)
        
    top_embedding_hit = search_results[0]
    query_feats = extract_local_features_from_array(img_crop_rgb)
    
    best_match = None
    min_inliers_valid = cfg['pipeline']['verification']['min_inliers_valid']
    max_inliers_found = 0 

    feat_type = cfg['models']['local_features']['type'].lower()
    valid_matches = []
    
    for hit in search_results:
        cand_feat_path = hit.payload.get("feature_path")
        
        if not cand_feat_path: continue
            
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
    
            num_matches = len(matches)
            if num_matches < cfg['pipeline']['verification']['min_matches_lightglue']: continue
        
            pts0 = query_feats['keypoints'][0][matches[:, 0]].cpu().numpy()
            pts1 = cand_feats['keypoints'][0][matches[:, 1]].cpu().numpy()
        
        H, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, cfg['pipeline']['verification']['ransac_threshold'])
        if mask is None: continue
            
        inliers = mask.sum()
        conf_ratio = inliers / num_matches if num_matches > 0 else 0
        is_geometric_valid = (inliers >= min_inliers_valid) or (inliers >= 10 and conf_ratio > 0.5)
        
        if inliers > max_inliers_found:
            max_inliers_found = inliers

        if is_geometric_valid:
            match_data = hit.payload.copy()
            match_data["color_hsv"] = hit.vector["color_hsv"] 
            match_data["inliers"] = int(inliers) 
            match_data["conf_ratio"] = float(conf_ratio) 
            match_data["fusion_score"] = float(hit.score)
            valid_matches.append(match_data)
                
    if valid_matches:
        valid_matches.sort(key=lambda x: x["inliers"], reverse=True)
        top_inliers = valid_matches[0]["inliers"]
        margin = max(5, int(top_inliers * 0.15))
        tied_candidates = [m for m in valid_matches if m["inliers"] >= (top_inliers - margin)]
        
        for cand in tied_candidates:
            cand_color = np.array(cand.get("color_hsv", np.zeros(128)))
            
            if np.sum(vec_query_color) == 0 or np.sum(cand_color) == 0:
                cand["color_score"] = 0.0
            else:
                cand["color_score"] = float(np.dot(vec_query_color, cand_color))
            
            cand["selection_score"] = cand["fusion_score"]
            
            if cand["color_score"] >= 0.45: 
                cand["selection_score"] *= 1.2
            elif cand["color_score"] < 0.25: 
                cand["selection_score"] *= 0.1 
                
        best_match = max(tied_candidates, key=lambda x: x["selection_score"])
   
        final_inliers = best_match["inliers"]
        final_score = best_match["fusion_score"]
        
        # 2. VETO DINÁMICO (Ajustado para escalas de Coseno vs RRF)
        best_match["verified"] = False 
        
        # 🚀 NUEVO: Detectamos si la escala de score es alta (Coseno crudo de Qwen/DINO) o baja (RRF Fusion)
               
        if search_mode in ["dinov2_only", "qwen_only"]:
            # Umbrales estrictos para Qwen/DINOv2 sin re-ranking
            thresh_high_inliers = 0.80 # Elimina el FP de Inliers=25, Score=0.791
            thresh_med_inliers = 0.85  # Elimina el FP de Inliers=15, Score=0.841
        else:
            # Umbrales originales para RRF Fusion
            thresh_high_inliers = 0.10
            thresh_med_inliers = 0.35

        if final_inliers >= 20:
            if final_score >= thresh_high_inliers:
                best_match["verified"] = True
                
        elif final_inliers >= 14:
            if final_score >= thresh_med_inliers: 
                best_match["verified"] = True       
        else:
            best_match["verified"] = False
        
        best_match["embedding"] = returned_embedding.tolist()    
        return best_match
        
    else:
        top_score = search_results[0].score
        margin = top_score * 0.05 
        
        fallback_candidates = [hit for hit in search_results if hit.score >= (top_score - margin)]
        
        best_fallback_hit = fallback_candidates[0]
        best_shadow_score = -1.0
        final_color_score = 0.0
        
        for hit in fallback_candidates:
            cand_color = np.array(hit.vector.get("color_hsv", np.zeros(128)))
            
            if np.sum(vec_query_color) == 0 or np.sum(cand_color) == 0:
                color_score = 0.0
            else:
                color_score = float(np.dot(vec_query_color, cand_color))
                
            shadow_score = hit.score
            
            if color_score >= 0.45: 
                shadow_score *= 1.2
            elif color_score < 0.25: 
                shadow_score *= 0.1
                
            if shadow_score > best_shadow_score:
                best_shadow_score = shadow_score
                best_fallback_hit = hit
                final_color_score = color_score

        fallback_match = best_fallback_hit.payload.copy()
        fallback_match["fusion_score"] = float(best_fallback_hit.score) 
        fallback_match["color_score"] = float(final_color_score)
        fallback_match["inliers"] = int(max_inliers_found)
        fallback_match["verified"] = False 

        fallback_match["embedding"] = returned_embedding.tolist()
        
        return fallback_match    