# Build context: raiz do repo (docker compose: dockerfile gateway/Dockerfile)

# --- Stage 1: build do frontend (Vite) ---------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* /build/
RUN npm ci || npm install
COPY frontend/ /build/
RUN npm run build

# --- Stage 2: runtime Python --------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt

COPY gateway/ /app/gateway/
COPY --from=frontend /build/dist /app/frontend/dist

RUN useradd -m app
USER app

EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
