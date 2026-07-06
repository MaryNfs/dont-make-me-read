from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from provider_config import get_embedding_dimension


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=None):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim or get_embedding_dimension()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )
        contexts = []
        sources = set()

        results = getattr(response, "points", []) or []
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def count_points(self) -> int:
        result = self.client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    def count_by_source(self, source: str) -> int:
        result = self.client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source),
                    )
                ]
            ),
            exact=True,
        )
        return int(result.count)
