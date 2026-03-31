import os
import uuid
import base64
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from celery import Celery
from celery.result import AsyncResult

# 1. Configuración de la App y Celery
app = FastAPI(
    title="Retail Re-ID API",
    description="API Gateway para el sistema de re-identificación visual SOTA",
    version="1.0.0"
)

# Permitir conexiones desde cualquier Web UI (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión al mismo Redis que usa el Worker
celery_client = Celery(
    'reid_api_client',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
)

# Asegurar que el buzón exista al arrancar
TMP_DIR = "/app/tmp_queries"
os.makedirs(TMP_DIR, exist_ok=True)

# 2. Modelos de Datos (Validación Pydantic)
class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Imagen original codificada en Base64")
    bboxes: list[list[float]] = Field(
        ..., 
        description="Lista de Bounding Boxes. Ejemplo: [[150, 200, 350, 450], [10, 20, 100, 150]]"
    )

# --- Añadir debajo de tu clase PredictRequest ---
class DetectRequest(BaseModel):
    image_b64: str = Field(..., description="Imagen codificada en Base64 para detección YOLO")

# 3. Endpoints REST
@app.post("/api/v1/predict", summary="Enviar imagen y BBoxes para clasificación")
async def predict_products(request: PredictRequest):
    """
    Recibe la imagen y los recortes. Guarda la imagen en un volumen compartido
    y envía SOLO la ruta física a la cola de la GPU para ahorrar RAM.
    """
    if not request.bboxes:
        raise HTTPException(status_code=400, detail="Debe proporcionar al menos un Bounding Box.")
    
    try:
        # 👈 NUEVO: Guardar en disco y generar ruta
        unique_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(TMP_DIR, unique_filename)
        
        img_data = base64.b64decode(request.image_b64)
        with open(image_path, "wb") as f:
            f.write(img_data)
            
        # 👈 NUEVO: Enviar 'image_path' en lugar de 'request.image_b64'
        task = celery_client.send_task(
            "tasks.predict_bboxes",
            args=[image_path, request.bboxes]
        )
        
        return {"message": "Trabajo encolado exitosamente", "task_id": task.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/api/v1/results/{task_id}", summary="Consultar el resultado de un trabajo")
async def get_results(task_id: str):
    """
    Consulta a Redis el estado de un task_id. 
    Puede devolver: PENDING, STARTED, SUCCESS, FAILURE.
    """
    task_result = AsyncResult(task_id, app=celery_client)
    
    if task_result.state == 'PENDING':
        return {"task_id": task_id, "status": "PENDING", "message": "En fila de espera..."}
    elif task_result.state == 'STARTED':
        return {"task_id": task_id, "status": "PROCESSING", "message": "La GPU está procesando los recortes..."}
    elif task_result.state == 'SUCCESS':
        return {"task_id": task_id, "status": "SUCCESS", "result": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"task_id": task_id, "status": "FAILED", "error": str(task_result.info)}
    else:
        return {"task_id": task_id, "status": task_result.state}

@app.post("/api/v1/catalog/sync", summary="Forzar sincronización del catálogo")
async def sync_catalog():
    """
    Dispara la tarea en segundo plano para leer el CSV y actualizar Qdrant.
    Útil cuando subes nuevas imágenes a la carpeta /data/Catalogo
    """
    task = celery_client.send_task("tasks.sync_catalog")
    return {"message": "Trabajo de sincronización de catálogo encolado", "task_id": task.id}

@app.get("/api/v1/image", summary="Obtener imagen de referencia")
async def get_reference_image(path: str):
    """
    Sirve la imagen estática de referencia desde el catálogo para la interfaz web.
    """
    # 1. Seguridad: Solo permitimos leer dentro de la carpeta del catálogo
    if not path.startswith("/app/data/Catalogo"):
        raise HTTPException(status_code=403, detail="Acceso denegado. Ruta no permitida.")
    
    # 2. Verificamos que el archivo exista
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="La imagen no fue encontrada en el servidor.")
        
    # 3. Devolvemos la imagen
    return FileResponse(path)

# 🚀 NUEVO ENDPOINT: Subida de Video y encolado en 'video_queue'
@app.post("/api/v1/video/analyze", summary="Subir video para conteo de stock")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    try:
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        video_path = os.path.join(TMP_DIR, unique_filename)

        with open(video_path, "wb") as f:
            f.write(await file.read())

        # Especificamos que vaya a la cola 'video_queue'
        task = celery_client.send_task(
            "tasks.process_video",
            args=[video_path],
            queue="video_queue"
        )

        return {"message": "Video encolado para procesamiento", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/v1/detect_bboxes", summary="Autodetectar BBoxes con YOLO")
async def auto_detect_bboxes(request: DetectRequest):
    """
    Guarda la imagen y envía la orden al worker de video para que use su YOLO.
    """
    try:
        unique_filename = f"yolo_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(TMP_DIR, unique_filename)
        
        img_data = base64.b64decode(request.image_b64)
        with open(image_path, "wb") as f:
            f.write(img_data)
            
        # 🚀 La magia: Lo enviamos explícitamente a la 'video_queue'
        task = celery_client.send_task(
            "tasks.detect_bboxes",
            args=[image_path],
            queue="video_queue" 
        )
        return {"message": "Auto-detección encolada", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))