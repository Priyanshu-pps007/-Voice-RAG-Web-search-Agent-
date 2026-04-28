import os
from pathlib import Path
from typing import Any
import uuid
from urllib.error import URLError

import ollama
from qdrant_client import models
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from RAG.config import client, collection_name, ensure_collection_ready

EMBEDDING_MODEL = "bge-m3"
DEFAULT_USER_ID = "priyanshu_pratap_singh"
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"


def ensure_collection_exists() -> None:
    try:
        ensure_collection_ready()
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to Qdrant. Check QDRANT_CLUSTER_ENDPOINT/QDRANT_API_KEY "
            "or make sure your local Qdrant server is running."
        ) from exc


def _extract_embedding_vector(response: Any) -> list[float]:
    if hasattr(response, "embeddings"):
        embeddings = response.embeddings
    elif isinstance(response, dict):
        embeddings = response.get("embeddings")
    else:
        embeddings = None

    if not embeddings or not isinstance(embeddings, list):
        raise ValueError("Embedding response did not include embeddings.")

    vector = embeddings[0]
    if not isinstance(vector, list):
        raise ValueError("Embedding vector is not a list.")

    return vector


def _pdf_chunks(file_path: str) -> list[dict[str, Any]]:
    try:
        elements = partition_pdf(filename=file_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to parse PDF '{os.path.basename(file_path)}'. "
            "Check your PDF parsing dependencies and API configuration."
        ) from exc

    chunks = chunk_by_title(
        elements,
        max_characters=500,
        new_after_n_chars=400,
        overlap=0,
    )

    prepared_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue

        prepared_chunks.append(
            {
                "text": text,
                "metadata": chunk.metadata.to_dict(),
            }
        )

    return prepared_chunks


def ingest_pdf(file_path: str, user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
    ensure_collection_exists()

    chunk_count = 0
    file_name = os.path.basename(file_path)

    for chunk in _pdf_chunks(file_path):
        try:
            response = ollama.embed(model=EMBEDDING_MODEL, input=chunk["text"])
        except Exception as exc:
            raise RuntimeError(
                "Unable to connect to Ollama for embeddings. Make sure Ollama is running "
                f"and the '{EMBEDDING_MODEL}' model is available."
            ) from exc

        vector = _extract_embedding_vector(response)

        try:
            client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=vector,
                        payload={
                            "user_id": user_id,
                            "text": chunk["text"],
                            "metadata": {
                                **chunk["metadata"],
                                "source_file": file_name,
                            },
                        },
                    )
                ],
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to store embeddings in Qdrant. Check Qdrant connectivity and credentials."
            ) from exc
        chunk_count += 1

    return {
        "file_name": file_name,
        "chunk_count": chunk_count,
        "user_id": user_id,
    }


def ingest_pdfs(file_paths: list[str], user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    total_chunks = 0

    for file_path in file_paths:
        result = ingest_pdf(file_path=file_path, user_id=user_id)
        results.append(result)
        total_chunks += result["chunk_count"]

    return {
        "user_id": user_id,
        "file_count": len(results),
        "total_chunks": total_chunks,
        "files": results,
    }


if __name__ == "__main__":
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = [str(path) for path in UPLOADS_DIR.glob("*.pdf")]
    print(ingest_pdfs(pdfs))
