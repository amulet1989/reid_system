import os
import cv2
import numpy as np
import torch 
import gc    
from celery import Celery
from celery.signals import worker_process_init
from PIL import Image, ImageOps

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
def predict_bboxes_task(image_path: str, bboxes: list): # 👈 NUEVO: Recibe image_path
    """
    Recibe la ruta física de una imagen temporal y una lista de Bounding Boxes.
    Lee la imagen del disco compartido, la procesa y elimina el archivo.
    """
    try:
        # 🚀 REEMPLAZO: Leemos con Pillow y horneamos el EXIF en lugar de cv2.imread
        try:
            pil_img = Image.open(image_path).convert("RGB")
            pil_img = ImageOps.exif_transpose(pil_img) # Hornear rotación física
            img_rgb = np.array(pil_img)
        except Exception as e:
            return {"error": f"No se pudo leer la imagen desde el volumen: {image_path}. Error: {e}"}
        
        results = []
        
        # Iterar sobre cada BBox detectado
        for i, bbox in enumerate(bboxes):
            img_crop = pipeline.crop_bbox(img_rgb, bbox)
            match = pipeline.query_online_multimodal_cached(img_crop)
            
            results.append({
                "bbox_index": i,
                "bbox_coords": bbox,
                "prediction": match
            })
            
        return {"status": "success", "detections": results}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # 🧹 PROTOCOLO DE LIMPIEZA EXTREMA V2 (Disco + VRAM)
        
        # 👈 NUEVO: Eliminar el archivo temporal para no saturar el SSD
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Warning: No se pudo eliminar {image_path}: {e}")

        # Limpiamos las variables temporales
        if 'img_bgr' in locals(): del img_bgr
        if 'img_rgb' in locals(): del img_rgb
        if 'img_crop' in locals(): del img_crop
        
        # Forzar al Garbage Collector
        gc.collect() 
        
        # Vaciar la caché de PyTorch
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