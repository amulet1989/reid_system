import os
from fastapi import FastAPI, HTTPException
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

# 2. Modelos de Datos (Validación Pydantic)
class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Imagen original codificada en Base64")
    bboxes: list[list[float]] = Field(
        ..., 
        description="Lista de Bounding Boxes. Ejemplo: [[150, 200, 350, 450], [10, 20, 100, 150]]"
    )

# 3. Endpoints REST
@app.post("/api/v1/predict", summary="Enviar imagen y BBoxes para clasificación")
async def predict_products(request: PredictRequest):
    """
    Recibe la imagen y los recortes, y los envía a la cola de la GPU.
    Devuelve un task_id inmediatamente (Latencia ~10ms).
    """
    if not request.bboxes:
        raise HTTPException(status_code=400, detail="Debe proporcionar al menos un Bounding Box.")
    
    # Enviar tarea a la cola (El nombre debe coincidir EXACTAMENTE con el del worker.py)
    task = celery_client.send_task(
        "tasks.predict_bboxes",
        args=[request.image_b64, request.bboxes]
    )
    
    return {"message": "Trabajo encolado exitosamente", "task_id": task.id}

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