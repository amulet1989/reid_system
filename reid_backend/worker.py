import os
import base64
import cv2
import numpy as np
import torch 
import gc    
from celery import Celery
from celery.signals import worker_process_init

# Importamos nuestro pipeline maestro
import pipeline

# Inicializar la aplicación Celery
# Lee las URLs de conexión desde el .env (o usa localhost por defecto si corres fuera de Docker)
celery_app = Celery(
    'reid_worker',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configuración adicional de Celery para manejar tareas pesadas
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    worker_prefetch_multiplier=1, # Que el worker solo tome 1 tarea a la vez (vital para VRAM)
    task_track_started=True
)

@worker_process_init.connect
def load_models_on_start(**kwargs):
    """
    Esta señal garantiza que PyTorch y CUDA se inicialicen de forma segura 
    únicamente dentro del proceso hijo de Celery.
    """
    pipeline.init_system()

@celery_app.task(name="tasks.predict_bboxes")
def predict_bboxes_task(image_b64: str, bboxes: list):
    """
    Recibe una imagen codificada en Base64 y una lista de Bounding Boxes.
    Decodifica la imagen en RAM y procesa cada recorte.
    bboxes format: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    try:
        # 1. Decodificar la imagen Base64 a NumPy Array en memoria (sin tocar el disco)
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return {"error": "No se pudo decodificar la imagen Base64."}
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        results = []
        
        # 2. Iterar sobre cada BBox detectado
        for i, bbox in enumerate(bboxes):
            # Recortar la imagen en memoria
            img_crop = pipeline.crop_bbox(img_rgb, bbox)
            
            # Pasar el recorte al pipeline multimodal
            match = pipeline.query_online_multimodal_cached(img_crop)
            
            # Guardamos el resultado asociado a su BBox original
            results.append({
                "bbox_index": i,
                "bbox_coords": bbox,
                "prediction": match
            })
            
        return {"status": "success", "detections": results}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # 🧹 PROTOCOLO DE LIMPIEZA EXTREMA (Se ejecuta SIEMPRE, al terminar o fallar)
        # Limpiamos las variables temporales gigantes si existen en el entorno local
        if 'img_bgr' in locals(): del img_bgr
        if 'img_rgb' in locals(): del img_rgb
        if 'img_crop' in locals(): del img_crop
        if 'nparr' in locals(): del nparr
        if 'img_data' in locals(): del img_data
        
        # 1. Forzar al Garbage Collector de Python a limpiar RAM del sistema
        gc.collect() 
        
        # 2. Vaciar la caché de PyTorch para liberar la VRAM de la GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@celery_app.task(name="tasks.sync_catalog")
def sync_catalog_task():
    """
    Tarea manual para forzar una re-ingesta del catálogo si el CSV o las imágenes cambiaron.
    """
    try:
        pipeline.batch_ingest_catalog()
        return {"status": "success", "message": "Sincronización del catálogo completada."}
    except Exception as e:
        return {"status": "error", "message": str(e)}