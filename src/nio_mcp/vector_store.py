import hashlib
import logging
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from nio_mcp.models import MessageRecord, SearchResult

logger = logging.getLogger(__name__)

VECTOR_SIZE = 1536  # text-embedding-3-small


def _event_id_to_uuid(event_id: str) -> str:
    digest = hashlib.sha256(event_id.encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest))


class VectorStore:
    def __init__(self, host: str, port: int, collection: str) -> None:
        self._client = AsyncQdrantClient(host=host, port=port)
        self._collection = collection

    async def init_collection(self, vector_size: int = VECTOR_SIZE) -> None:
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", self._collection)
        else:
            logger.debug("Qdrant collection %s already exists", self._collection)

    async def upsert(self, record: MessageRecord, vector: list[float]) -> None:
        point = qmodels.PointStruct(
            id=_event_id_to_uuid(record.event_id),
            vector=vector,
            payload={
                "event_id": record.event_id,
                "room_id": record.room_id,
                "sender": record.sender,
                "sender_name": record.sender_name,
                "body": record.body,
                "timestamp": record.timestamp,
            },
        )
        await self._client.upsert(
            collection_name=self._collection,
            points=[point],
            wait=False,
        )

    async def search(
        self,
        vector: list[float],
        limit: int = 10,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> list[SearchResult]:
        query_filter = None
        conditions: list[qmodels.FieldCondition] = []
        if room_id:
            conditions.append(
                qmodels.FieldCondition(key="room_id", match=qmodels.MatchValue(value=room_id))
            )
        if sender:
            conditions.append(
                qmodels.FieldCondition(key="sender", match=qmodels.MatchValue(value=sender))
            )
        if conditions:
            query_filter = qmodels.Filter(must=conditions)

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        results = []
        for hit in hits:
            p = hit.payload
            results.append(
                SearchResult(
                    event_id=p["event_id"],
                    room_id=p["room_id"],
                    sender=p["sender"],
                    sender_name=p.get("sender_name", p["sender"]),
                    body=p["body"],
                    timestamp=p["timestamp"],
                    score=hit.score,
                )
            )
        return results

    async def close(self) -> None:
        await self._client.close()
