"""Ingestão offline do corpus de políticas (docs/policies/*.md) no Qdrant.

Chunk por seção (gateway.document_search.chunk_markdown) → SBERT → upsert
idempotente (id = hash do conteúdo) na collection `documents`. Re-rodar após
qualquer edição nas políticas.

Uso:
    set -a; source .env; set +a
    SBERT_CACHE_DIR=$PWD/models .venv/bin/python scripts/ingest_documents.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import load_settings  # noqa: E402
from gateway.document_search import DOCUMENTS_COLLECTION, chunk_id, chunk_markdown  # noqa: E402
from gateway.embedder import SBERTEmbedder  # noqa: E402

POLICIES_DIR = ROOT / "docs" / "policies"


def main() -> int:
    settings = load_settings()
    embedder = SBERTEmbedder(model_name=settings.sbert_model, cache_dir=settings.sbert_cache_dir)
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    client = httpx.Client(timeout=30.0, headers=headers)

    chunks = []
    for path in sorted(POLICIES_DIR.glob("*.md")):
        file_chunks = chunk_markdown(path.read_text(encoding="utf-8"), document=path.stem)
        chunks.extend(file_chunks)
        print(f"{path.name}: {len(file_chunks)} chunks")
    if not chunks:
        print("Nenhum documento em docs/policies/.", file=sys.stderr)
        return 2

    started = time.monotonic()
    vectors = embedder.embed([c["text"] for c in chunks], prefix_type="document")

    response = client.get(f"{settings.qdrant_url}/collections/{DOCUMENTS_COLLECTION}")
    if response.status_code != 200:
        client.put(
            f"{settings.qdrant_url}/collections/{DOCUMENTS_COLLECTION}",
            json={"vectors": {"size": embedder.dim, "distance": "Cosine"}},
        ).raise_for_status()

    points = [
        {"id": chunk_id(chunk), "vector": vector, "payload": chunk}
        for chunk, vector in zip(chunks, vectors)
    ]
    client.put(
        f"{settings.qdrant_url}/collections/{DOCUMENTS_COLLECTION}/points?wait=true",
        json={"points": points},
    ).raise_for_status()
    print(f"\n{len(points)} chunks indexados em '{DOCUMENTS_COLLECTION}' "
          f"({time.monotonic() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
