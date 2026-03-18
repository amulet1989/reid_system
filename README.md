# 🛒 Motor de Re-Identificación Visual para Retail (SOTA)

Este proyecto es un microservicio asíncrono de Inteligencia Artificial diseñado para la re-identificación de productos en estanterías de retail. Utiliza una arquitectura multimodal State-of-the-Art (SOTA) que combina análisis semántico, OCR y geometría epipolar para lograr una precisión extrema.

## 🧠 Arquitectura de Modelos
El motor fusiona múltiples enfoques de visión computacional:
1. **DINOv2 (Meta):** Extracción de características visuales densas (texturas, formas, colores).
2. **Qwen3-VL (Alibaba):** Análisis de diseño (layout) y extracción de texto/marcas (OCR) en superficies curvas.
3. **ALIKED + LightGlue:** Extracción y emparejamiento de keypoints geométricos locales.
4. **Homografía RANSAC:** Validación espacial 2D estricta para descartar falsos positivos.
5. **Qdrant:** Base de datos vectorial con fusión de scores (RRF - Reciprocal Rank Fusion).

## ⚙️ Arquitectura del Sistema (Worker-Queue)
Para evitar bloqueos (OOM) en la GPU (ej. RTX 3080 Ti), el sistema utiliza un patrón de colas:
* **FastAPI (API Gateway):** Recibe las peticiones HTTP (CPU-bound) y las encola en Redis. Retorna una respuesta instantánea.
* **Redis (Message Broker):** Gestiona la cola de trabajos.
* **Celery (GPU Worker):** Mantiene los modelos neuronales precargados en la VRAM. Toma los trabajos de Redis uno por uno, procesa los recortes en memoria RAM y devuelve los resultados.

---

## 🚀 Guía de Integración (API Reference)

Para integrar este motor de inferencia a tu aplicación, solo necesitas interactuar con la API REST (FastAPI) expuesta en el puerto `8000`. Todo el manejo de Redis y Celery es transparente para el cliente.

### 1. Enviar un trabajo de Inferencia
Envía la imagen original completa y las coordenadas de dónde están los productos. La API no procesa la imagen, solo genera un "Ticket de trabajo".

* **Endpoint:** `POST /api/v1/predict`
* **Content-Type:** `application/json`
* **Payload (Body):**
```json
{
  "image_b64": "/9j/4AAQSkZJRgABAQEAAAAAA...", 
  "bboxes": [
    [150, 200, 350, 450], 
    [400, 210, 600, 460]
  ]
}
```

Nota: Las coordenadas deben estar en formato [x1, y1, x2, y2] (top-left, bottom-right).

Respuesta Exitosa (200 OK):
```json
{
  "message": "Trabajo encolado exitosamente",
  "task_id": "ebc4be82-6327-4e82-9324-b4e5e6749580"
}
```
### 2. Consultar el Resultado (Polling)
Usa el task_id obtenido para consultar el estado del trabajo. Se recomienda hacer polling cada 0.5 - 1.0 segundos.

- Endpoint: GET /api/v1/results/{task_id}

- Estados posibles de respuesta: PENDING, PROCESSING, SUCCESS, FAILED.

- Respuesta cuando el trabajo está listo (SUCCESS):

```json
{
  "task_id": "ebc4be82-6327-4e82-9324-b4e5e6749580",
  "status": "SUCCESS",
  "result": {
    "status": "success",
    "detections": [
      {
        "bbox_index": 0,
        "bbox_coords": [150, 200, 350, 450],
        "prediction": {
          "sku": "604722006137",
          "name": "CACAHUATE JAPONES LEO 90GRS",
          "view": "IMG_0650_crop",
          "fusion_score": 0.985,
          "inliers": 45,
          "verified": true
        }
      },
      {
        "bbox_index": 1,
        "bbox_coords": [400, 210, 600, 460],
        "prediction": {
          "sku": "SKU-DESCONOCIDO",
          "name": "Producto Similar",
          "fusion_score": 0.450,
          "inliers": 2,
          "verified": false
        }
      }
    ]
  }
}
```
### 💡 Cómo interpretar prediction en tu aplicación:

Si verified: true: RANSAC confirmó la topología. Confianza absoluta (100%).

Si verified: false: La IA Semántica (embeddings) encontró similitudes, pero la geometría falló (ej. producto nuevo, ángulo extremo o imagen borrosa).

### 3. Sincronizar Catálogo (Administración)
Si agregas nuevas imágenes a la carpeta data/Catalogo o modificas el archivo data/Catalogo.csv, ejecuta este endpoint para actualizar la base de datos vectorial Qdrant y la caché sin reiniciar el servidor.

- Endpoint: POST /api/v1/catalog/sync

- Respuesta:

```
{
  "message": "Trabajo de sincronización de catálogo encolado",
  "task_id": "c7a2b91d-..."
}
```
### 🛠️ Despliegue Local
1. Clona el repositorio.

2. Asegúrate de tener instalado docker, docker-compose y nvidia-container-toolkit.

3. Coloca tus imágenes base y el CSV en la carpeta /data.

4. Levanta la arquitectura completa:
```
docker-compose up -d --build
```
El sistema expondrá:

- API Gateway: http://localhost:8000

- Panel Web UI: http://localhost:8501

- Qdrant DB: http://localhost:6333