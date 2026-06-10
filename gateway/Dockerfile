# Build context: raiz do repo (docker compose: dockerfile gateway/Dockerfile)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt

COPY gateway/ /app/gateway/

EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
