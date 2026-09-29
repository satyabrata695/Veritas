FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements-server.txt /app/requirements-server.txt
# Render has no GPU. CPU-only PyTorch avoids downloading unused CUDA packages.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r /app/requirements-server.txt

COPY backend /app/backend
COPY models /app/models
COPY frontend /app/frontend

WORKDIR /app/backend

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
