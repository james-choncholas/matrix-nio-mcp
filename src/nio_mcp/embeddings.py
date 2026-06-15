import asyncio
import logging
from typing import Optional

import tiktoken
from openai import AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)

_FALLBACK_ENCODING = "cl100k_base"
_MAX_RETRIES = 3


class EmbeddingError(Exception):
    pass


class EmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        max_tokens: int = 8191,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._max_tokens = max_tokens
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)

    def _truncate(self, text: str) -> str:
        tokens = self._encoding.encode_ordinary(text)
        safe_max = min(self._max_tokens, 8191)
        if len(tokens) <= safe_max:
            return text

        # When truncating, truncate to slightly less than safe_max (leaving a buffer)
        # to prevent partial/malformed UTF-8 characters at the boundary from expanding
        # when decoded and re-encoded.
        truncate_len = max(1, safe_max - 16)
        logger.warning("Truncating text from %d to %d tokens", len(tokens), truncate_len)
        return self._encoding.decode(tokens[:truncate_len])

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        truncated = [self._truncate(t) for t in texts]
        kwargs = {"input": truncated, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.embeddings.create(**kwargs)
                items = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in items]
            except RateLimitError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limited by OpenAI; retrying in %.0fs (attempt %d/%d)",
                        delay, attempt + 1, _MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
            except Exception as exc:
                raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc

        raise EmbeddingError(
            f"OpenAI embedding failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc
