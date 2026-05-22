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

        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="timestamp",
            field_schema=qmodels.IntegerIndexParams(
                type=qmodels.IntegerIndexType.INTEGER,
                is_principal=True,
            ),
        )

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

    def _build_filter(
        self,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> Optional[qmodels.Filter]:
        conditions: list[qmodels.FieldCondition] = []
        if room_id:
            conditions.append(
                qmodels.FieldCondition(key="room_id", match=qmodels.MatchValue(value=room_id))
            )
        if sender:
            conditions.append(
                qmodels.FieldCondition(key="sender", match=qmodels.MatchValue(value=sender))
            )
        if after_ts is not None or before_ts is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="timestamp",
                    range=qmodels.Range(
                        gte=after_ts,
                        lte=before_ts,
                    ),
                )
            )
        return qmodels.Filter(must=conditions) if conditions else None

    async def search(
        self,
        vector: list[float],
        limit: int = 10,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> list[SearchResult]:
        query_filter = self._build_filter(room_id, sender, after_ts, before_ts)
        if hasattr(self._client, "search"):
            hits = await self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            hits = response.points
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

    async def scroll(
        self,
        limit: int = 10,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> list[SearchResult]:
        points, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._build_filter(room_id, sender, after_ts, before_ts),
            limit=limit,
            order_by=qmodels.OrderBy(
                key="timestamp",
                direction=qmodels.Direction.DESC,
            ),
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for point in points:
            p = point.payload
            results.append(
                SearchResult(
                    event_id=p["event_id"],
                    room_id=p["room_id"],
                    sender=p["sender"],
                    sender_name=p.get("sender_name", p["sender"]),
                    body=p["body"],
                    timestamp=p["timestamp"],
                    score=0.0,
                )
            )
        return results

    async def close(self) -> None:
        await self._client.close()
