import os
import uuid
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import torchvision.transforms as T
from transformers import AutoProcessor, AutoModel
from lightglue import LightGlue, ALIKED
from lightglue.utils import numpy_image_to_torch, rbd
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

    print("🧠 Cargando ALIKED y LightGlue...")
    extractor = ALIKED(
        max_num_keypoints=cfg['models']['aliked']['max_num_keypoints'], 
        detection_threshold=cfg['models']['aliked']['detection_threshold']
    ).eval().to(DEVICE)
    matcher = LightGlue(features='aliked').eval().to(DEVICE)

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
                "qwen_layout": VectorParams(size=cfg['models']['qwen']['vector_size'], distance=Distance.COSINE)
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
        
    raw_prompt = f"Extract all text, brand names, and analyze the layout of this retail product packaging. Context: {metadata_text}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": raw_prompt}]}]
    
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(images=image, text=text_prompt, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.inference_mode():
        outputs = qwen_model(**inputs)
        emb = outputs.last_hidden_state[:, -1, :] 
    
    emb = emb.float()
    return (emb / emb.norm(dim=-1, keepdim=True))[0].cpu().numpy()

def extract_aliked_features_from_array(img_rgb):
    img = img_rgb.copy()
    max_dim = cfg['models']['aliked']['max_image_size']
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    image_tensor = numpy_image_to_torch(img).to(DEVICE)
    with torch.inference_mode():
        feats = extractor.extract(image_tensor)
    return feats

# --- 4. FUNCIONES DE INGESTA (OFFLINE) ---
def index_product_offline(sku, name, category, image_path, view_name="frente"):
    
    # img_bgr = cv2.imread(image_path)
    # if img_bgr is None:
    #     raise ValueError(f"No se pudo leer la imagen: {image_path}")
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        pil_img = Image.open(image_path).convert("RGB")
        # --- PARCHE DE ROTACIÓN EXIF ---
        pil_img = ImageOps.exif_transpose(pil_img) # Hornear rotación física
        # -------------------------------
        img_rgb = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"No se pudo leer la imagen con Pillow: {image_path}. Error: {e}")

    vec_visual = get_dinov2_embedding_from_array(img_rgb)
    context = f"Product: {name}. Category: {category}. Packaging view: {view_name}."
    vec_layout = get_qwen_layout_embedding_from_array(img_rgb, metadata_text=context)
    
    feats = extract_aliked_features_from_array(img_rgb)
    feats_cpu = {k: v.cpu() for k, v in feats.items() if isinstance(v, torch.Tensor)}
    
    safe_img_name = os.path.basename(image_path).replace(".", "_")
    feat_path = os.path.join(CACHE_DIR, f"{sku}_{safe_img_name}_aliked.pt")
    torch.save(feats_cpu, feat_path)
    
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=point_id, 
            vector={"dinov2": vec_visual.tolist(), "qwen_layout": vec_layout.tolist()}, 
            payload={
                "sku": sku, 
                "name": name, 
                "view": view_name,
                "image_path": image_path,
                "aliked_path": feat_path 
            }
        )]
    )

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

def query_online_multimodal_cached(img_crop_rgb):
    vec_visual = get_dinov2_embedding_from_array(img_crop_rgb)
    vec_layout = get_qwen_layout_embedding_from_array(img_crop_rgb, metadata_text="Find matching retail packaging.")
    
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
        
    top_embedding_hit = search_results[0]
    query_feats = extract_aliked_features_from_array(img_crop_rgb)
    
    best_match = None
    min_inliers_valid = cfg['pipeline']['verification']['min_inliers_valid']
    max_inliers = min_inliers_valid - 1 
    
    for hit in search_results:
        cand_feat_path = hit.payload["aliked_path"]
        try:
            cand_feats = torch.load(cand_feat_path, map_location=DEVICE, weights_only=False)
        except Exception:
            continue

        with torch.inference_mode():
            matches01 = matcher({"image0": query_feats, "image1": cand_feats})
            
        matches01 = rbd(matches01)
        matches = matches01['matches'] 
        
        if len(matches) < cfg['pipeline']['verification']['min_matches_lightglue']:
            continue
            
        pts0 = query_feats['keypoints'][0][matches[..., 0]].cpu().numpy()
        pts1 = cand_feats['keypoints'][0][matches[..., 1]].cpu().numpy()
        
        H, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, cfg['pipeline']['verification']['ransac_threshold'])
        if mask is None:
            continue
            
        inliers = mask.sum()
        
        if inliers > max_inliers:
            max_inliers = inliers
            best_match = hit.payload.copy()
            best_match["inliers"] = int(inliers)
            best_match["fusion_score"] = float(hit.score)
            
    if best_match:
        best_match["verified"] = True
        return best_match
    else:
        fallback_match = top_embedding_hit.payload.copy()
        fallback_match["fusion_score"] = float(top_embedding_hit.score)
        fallback_match["inliers"] = int(max_inliers) if max_inliers > 0 else 0 
        fallback_match["verified"] = False
        return fallback_match