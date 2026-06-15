import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nio_mcp.embeddings import EmbeddingClient, EmbeddingError


def _make_embedding_response(vectors: list[list[float]]):
    items = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.index = i
        item.embedding = vec
        items.append(item)
    resp = MagicMock()
    resp.data = items
    return resp


@pytest.fixture
def mock_openai():
    with patch("nio_mcp.embeddings.AsyncOpenAI") as cls:
        instance = cls.return_value
        instance.embeddings = MagicMock()
        instance.embeddings.create = AsyncMock()
        yield instance


@pytest.fixture
def client(mock_openai):
    return EmbeddingClient(api_key="test-key")


async def test_embed_returns_single_vector(client, mock_openai):
    vec = [0.1, 0.2, 0.3]
    mock_openai.embeddings.create.return_value = _make_embedding_response([vec])
    result = await client.embed("hello")
    assert result == vec


async def test_embed_batch_returns_all_vectors(client, mock_openai):
    vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    mock_openai.embeddings.create.return_value = _make_embedding_response(vecs)
    result = await client.embed_batch(["a", "b", "c"])
    assert result == vecs


async def test_embed_batch_empty_returns_empty(client, mock_openai):
    result = await client.embed_batch([])
    mock_openai.embeddings.create.assert_not_called()
    assert result == []


async def test_embed_batch_sorts_by_index(client, mock_openai):
    # API returns out-of-order items
    item0 = MagicMock()
    item0.index = 0
    item0.embedding = [1.0]
    item1 = MagicMock()
    item1.index = 1
    item1.embedding = [2.0]
    resp = MagicMock()
    resp.data = [item1, item0]  # reversed
    mock_openai.embeddings.create.return_value = resp

    result = await client.embed_batch(["first", "second"])
    assert result == [[1.0], [2.0]]


async def test_embed_raises_embedding_error_on_api_failure(client, mock_openai):
    mock_openai.embeddings.create.side_effect = RuntimeError("network error")
    with pytest.raises(EmbeddingError, match="network error"):
        await client.embed("hello")


async def test_embed_batch_passes_correct_model(client, mock_openai):
    mock_openai.embeddings.create.return_value = _make_embedding_response([[0.0]])
    await client.embed("test")
    call_kwargs = mock_openai.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "text-embedding-3-small"


async def test_embed_batch_passes_dimensions_when_set(mock_openai):
    mock_openai.embeddings.create.return_value = _make_embedding_response([[0.0] * 512])
    client = EmbeddingClient(api_key="test-key", dimensions=512)
    await client.embed("test")
    call_kwargs = mock_openai.embeddings.create.call_args
    assert call_kwargs.kwargs["dimensions"] == 512


async def test_embed_batch_omits_dimensions_when_not_set(client, mock_openai):
    mock_openai.embeddings.create.return_value = _make_embedding_response([[0.0]])
    await client.embed("test")
    call_kwargs = mock_openai.embeddings.create.call_args
    assert "dimensions" not in call_kwargs.kwargs


async def test_truncate_logic_no_truncation(client):
    # Short text should not be truncated
    text = "hello world"
    result = client._truncate(text)
    assert result == text


async def test_truncate_logic_truncates_long_text():
    # Setup client with max_tokens of 100
    client = EmbeddingClient(api_key="test-key", max_tokens=100)

    # We want a text that encodes to > 100 tokens. Let's make one.
    encoding = client._encoding
    long_text = " ".join(["token"] * 150)
    tokens = encoding.encode_ordinary(long_text)
    assert len(tokens) > 100

    truncated = client._truncate(long_text)
    truncated_tokens = encoding.encode_ordinary(truncated)
    # It should have been truncated to 100 - 16 = 84 tokens
    assert len(truncated_tokens) == 84


async def test_truncate_logic_respects_hard_safe_max():
    # Setup client with max_tokens of 10000 (which is larger than the hard 8191 limit)
    client = EmbeddingClient(api_key="test-key", max_tokens=10000)

    # Text with > 8191 tokens
    long_text = " ".join(["hello"] * 9000)
    encoding = client._encoding
    tokens = encoding.encode_ordinary(long_text)
    assert len(tokens) > 8191

    truncated = client._truncate(long_text)
    truncated_tokens = encoding.encode_ordinary(truncated)
    # The absolute maximum is 8191. So safe_max = min(10000, 8191) = 8191.
    # It should be truncated to 8191 - 16 = 8175 tokens
    assert len(truncated_tokens) == 8175
