FROM python:3.12-slim

ARG SERVICE
ENV SERVICE=${SERVICE} \
    DB_PATH=/data/${SERVICE}.db \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115,<1" "uvicorn>=0.34,<1"

COPY common.py ./common.py
COPY ${SERVICE}/ ./${SERVICE}/

EXPOSE 8000
CMD uvicorn ${SERVICE}.main:app --host 0.0.0.0 --port 8000
