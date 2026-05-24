import logging
from typing import Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class EmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        try:
            response = await self._client.embeddings.create(**kwargs)
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc

        # API returns items sorted by index
        items = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in items]
