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
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/gateway/requirements.txt

# Pre-download SBERT model (cached no image, evita download no primeiro request)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder='/app/models')"

COPY gateway/ /app/gateway/
# Golden de roteamento: fonte do índice do semantic router (Qdrant).
COPY evals/golden_routing.jsonl /app/evals/golden_routing.jsonl
# Dataset de injection para treino offline (não usado em runtime).
COPY train/injection_dataset.jsonl /app/train/injection_dataset.jsonl
COPY --from=frontend /build/dist /app/frontend/dist

RUN useradd -m app
USER app

EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
