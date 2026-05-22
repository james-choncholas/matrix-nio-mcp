import hashlib
import logging
import re
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


def _sender_search_text(sender: str, sender_name: str) -> str:
    values: list[str] = []
    seen: set[str] = set()

    def add(value: Optional[str]) -> None:
        if not value:
            return
        normalized = value.strip()
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        values.append(normalized)

    add(sender_name)
    add(sender)
    if sender.startswith("@"):
        localpart = sender[1:].split(":", 1)[0]
        add(localpart)
        add(localpart.replace(".", " ").replace("_", " ").replace("-", " "))

    return " ".join(values)


def _sender_query_terms(sender_query: str) -> list[str]:
    normalized = sender_query.strip().casefold()
    if not normalized:
        return []
    normalized = normalized.replace("@", " ").replace(":", " ")
    normalized = normalized.replace(".", " ").replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    terms: list[str] = []
    seen: set[str] = set()
    for term in normalized.split(" "):
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _looks_like_mxid(value: str) -> bool:
    return value.startswith("@") and ":" in value and " " not in value


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
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="sender_search",
            field_schema=qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.WORD,
                lowercase=True,
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
                "sender_search": _sender_search_text(record.sender, record.sender_name),
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
        sender_query: Optional[str] = None,
        sender_match_count: Optional[int] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> Optional[qmodels.Filter]:
        conditions: list[qmodels.FieldCondition] = []
        min_should: Optional[qmodels.MinShould] = None
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
        if sender_query:
            stripped_query = sender_query.strip()
            if _looks_like_mxid(stripped_query):
                conditions.append(
                    qmodels.FieldCondition(
                        key="sender",
                        match=qmodels.MatchValue(value=stripped_query),
                    )
                )
            else:
                sender_conditions = [
                    qmodels.FieldCondition(
                        key="sender_search",
                        match=qmodels.MatchText(text=term),
                    )
                    for term in _sender_query_terms(stripped_query)
                ]
                if sender_conditions:
                    min_count = sender_match_count or len(sender_conditions)
                    min_should = qmodels.MinShould(
                        conditions=sender_conditions,
                        min_count=max(1, min(min_count, len(sender_conditions))),
                    )

        if not conditions and min_should is None:
            return None
        return qmodels.Filter(must=conditions or None, min_should=min_should)

    def _candidate_filters(
        self,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
        sender_query: Optional[str] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> list[Optional[qmodels.Filter]]:
        if not sender_query or _looks_like_mxid(sender_query.strip()):
            return [
                self._build_filter(
                    room_id=room_id,
                    sender=sender,
                    sender_query=sender_query,
                    after_ts=after_ts,
                    before_ts=before_ts,
                )
            ]

        terms = _sender_query_terms(sender_query)
        if len(terms) <= 1:
            return [
                self._build_filter(
                    room_id=room_id,
                    sender=sender,
                    sender_query=sender_query,
                    sender_match_count=1,
                    after_ts=after_ts,
                    before_ts=before_ts,
                )
            ]

        return [
            self._build_filter(
                room_id=room_id,
                sender=sender,
                sender_query=sender_query,
                sender_match_count=len(terms),
                after_ts=after_ts,
                before_ts=before_ts,
            ),
            self._build_filter(
                room_id=room_id,
                sender=sender,
                sender_query=sender_query,
                sender_match_count=1,
                after_ts=after_ts,
                before_ts=before_ts,
            ),
        ]

    @staticmethod
    def _results_from_hits(hits) -> list[SearchResult]:
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

    async def _search_once(
        self,
        vector: list[float],
        limit: int,
        query_filter: Optional[qmodels.Filter],
    ) -> list[SearchResult]:
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
        return self._results_from_hits(hits)

    async def _merge_searches(
        self,
        filters: list[Optional[qmodels.Filter]],
        search_fn,
        limit: int,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        seen_event_ids: set[str] = set()

        for query_filter in filters:
            hits = await search_fn(limit, query_filter)
            for hit in hits:
                if hit.event_id in seen_event_ids:
                    continue
                seen_event_ids.add(hit.event_id)
                results.append(hit)
                if len(results) >= limit:
                    return results

        return results

    async def search(
        self,
        vector: list[float],
        limit: int = 10,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
        sender_query: Optional[str] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> list[SearchResult]:
        filters = self._candidate_filters(room_id, sender, sender_query, after_ts, before_ts)
        return await self._merge_searches(
            filters,
            lambda current_limit, query_filter: self._search_once(vector, current_limit, query_filter),
            limit,
        )

    async def _scroll_once(
        self,
        limit: int,
        scroll_filter: Optional[qmodels.Filter],
    ) -> list[SearchResult]:
        points, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=scroll_filter,
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

    async def scroll(
        self,
        limit: int = 10,
        room_id: Optional[str] = None,
        sender: Optional[str] = None,
        sender_query: Optional[str] = None,
        after_ts: Optional[int] = None,
        before_ts: Optional[int] = None,
    ) -> list[SearchResult]:
        filters = self._candidate_filters(room_id, sender, sender_query, after_ts, before_ts)
        return await self._merge_searches(filters, self._scroll_once, limit)

    async def close(self) -> None:
        await self._client.close()
