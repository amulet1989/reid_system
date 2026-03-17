# Usamos una imagen oficial de PyTorch con CUDA 12.1 (Compatible con la serie RTX 3000)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para OpenCV y Git (para descargar LightGlue)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar primero el requirements para aprovechar la caché de capas de Docker
COPY requirements.txt .

# Instalar las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente del backend
COPY . .

# Comando de inicio del Worker de Celery
# VITAL: --concurrency=1 asegura que solo haya un proceso cargando la GPU a la vez para evitar OOM (Out Of Memory)
CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--pool=solo"]