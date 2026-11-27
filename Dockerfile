# ==============================================================================
# Stage 1: Build React + Vite Production Frontend
# ==============================================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Python Flask + Scikit-Learn ML Runtime
# ==============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PORT=10000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY dataset/ ./dataset/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Train the reproducible ML pipeline natively inside this container's Python/sklearn
# environment (seed=42) to guarantee binary compatibility and genuine artifacts.
RUN mkdir -p models reports/figures && python backend/ml/train.py

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 --threads 2 --timeout 120 backend.app:app"]
