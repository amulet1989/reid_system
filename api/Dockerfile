# Usamos una imagen de Python ligera
FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc y forzar el log en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Arrancar el servidor web Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]