# =========================================================================
# Dockerfile - API de Riesgo Crediticio (PIM5)
# =========================================================================
# Construye una imagen autocontenida que ejecuta la API de FastAPI para
# servir predicciones del modelo de riesgo crediticio.
#
# Construir la imagen (desde la raíz del proyecto):
#     docker build -t riesgo-crediticio-api .
#
# Ejecutar el contenedor:
#     docker run -p 8000:8000 riesgo-crediticio-api
#
# La API quedará disponible en http://localhost:8000
# =========================================================================

# Imagen base ligera de Python
FROM python:3.11-slim

# Variables de entorno para un comportamiento limpio de Python en contenedor:
# - PYTHONDONTWRITEBYTECODE: no genera archivos .pyc
# - PYTHONUNBUFFERED: los logs salen inmediatamente (sin buffer)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para algunas librerías
# (libgomp1 es requerida por LightGBM y XGBoost).
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copiar primero requirements.txt para aprovechar la caché de Docker:
# si el código cambia pero las dependencias no, no se reinstala todo.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto (el .dockerignore excluye lo innecesario)
COPY . .

# Exponer el puerto de la API
EXPOSE 8000

# Comando de arranque: servir la API con uvicorn.
# Se usa 0.0.0.0 para que sea accesible desde fuera del contenedor.
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
