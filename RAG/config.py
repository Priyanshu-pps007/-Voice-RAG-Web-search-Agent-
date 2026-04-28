import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT") or "http://localhost:6333"
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

collection_name = "RAG_Collection"


def ensure_collection_ready() -> None:
    collections = client.get_collections().collections
    collection_names = {collection.name for collection in collections}

    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="user_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )




if __name__ == "__main__":
    ensure_collection_ready()
