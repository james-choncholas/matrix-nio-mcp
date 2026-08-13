# LLM webhook failure investigation

Date investigated: 2026-08-04 (UTC)

## Executive summary

The recurring webhook failures are primarily **client-side read timeouts in `nio-mcp`**, not a permanently invalid URL, bearer token, model, or tool configuration.

`WebhookDispatcher` creates its `httpx.AsyncClient` with a hard-coded **30-second timeout**. The configured OpenWebUI model (`steven`) is an agentic Anthropic-backed pipeline that loads memory, connects to MCP servers, may run tools, and can make multiple model calls before returning a non-streaming `/chat/completions` response. A meaningful fraction of those requests take longer than 30 seconds, so `nio-mcp` raises `httpx.ReadTimeout` while OpenWebUI is still working.

The failure handling then makes the impact worse: the batch is removed from `_pending_records` before the HTTP request and is neither requeued nor journaled when the request fails. Consequently, every timeout permanently drops that webhook batch from LLM processing.

## Evidence

### Deployment and request shape

The deployment is `nio-mcp` in namespace `matrix`. The checked-in `homekube/matrix/nio-mcp/deployment.yaml` configures:

- `WEBHOOK_URL=https://ai.clerb.club/api/v1`
- `WEBHOOK_MODEL=steven`
- an agent/tool set including Matrix, BookStack, email, calendar/Nextcloud, maps, and notification tools
- `WEBHOOK_COOLDOWN_SECONDS=1800`

The code appends `/chat/completions` and sends a conventional non-streaming OpenAI-compatible request. Thus the actual target is `https://ai.clerb.club/api/v1/chat/completions`.

### `nio-mcp` logs

Grafana Loki logs for `{namespace="matrix", pod=~"nio-mcp-.*", container="nio-mcp"}` repeatedly show:

```text
WARNING:nio_mcp.webhook:LLM webhook call failed for N buffered message(s); continuing
...
httpcore.ReadTimeout
...
httpx.ReadTimeout
```

Recent examples include 2026-08-03 01:09, 03:08, 13:05, and 13:43 UTC, and 2026-08-04 13:22 UTC. The failures affect batches ranging from one message to many messages (one unusually large batch contained 99 messages).

Over the 30-day window queried, Loki reported:

- **64** `LLM webhook call failed` events
- **138** completed HTTP `200 OK` responses from the same `/api/v1/chat/completions` endpoint

The successful calls demonstrate that the current URL, credential, model name, and request format are generally valid. This is an intermittent duration/reliability problem rather than a permanently broken configuration.

### Correlation with upstream OpenWebUI logs

The 2026-08-04 failure provides a clear correlated example:

- 13:22:19.527 UTC: OpenWebUI receives the first-turn request and begins downloading the `steven` agent's workspace/memory files.
- 13:22:20 UTC: it connects to Matrix and BookStack MCP servers and converts a large tool catalog.
- 13:22:23.514 UTC: its first Anthropic request completes.
- 13:22:43.224 UTC: tool-loop iteration 1 completes with a tool result.
- 13:22:45.406 UTC: a second Anthropic request completes.
- **13:22:49.339 UTC: `nio-mcp` reports `httpx.ReadTimeout`, almost exactly 30 seconds after the upstream request began.**
- 13:22:57 UTC: OpenWebUI is still cleaning up/disconnecting its Matrix MCP stream.

This aligns precisely with the hard-coded timeout and confirms that the upstream agent was active, not unreachable.

A second large-batch example on 2026-08-02 started upstream work at 16:58:28 UTC, made an Anthropic call at 16:58:32, and did not finish within the caller's 30-second window; `nio-mcp` logged failure at 16:58:58 UTC.

### Source-code behavior

In `src/nio_mcp/webhook.py`:

- Lines 69-70 create `httpx.AsyncClient(timeout=30.0)`; the fallback construction at lines 119-120 uses the same value.
- Lines 105-106 move all pending records out of `_pending_records` before calling the LLM.
- Lines 109-116 catch and log every exception but do not retry or restore the records.
- The live-message pending-index journal does not protect this stage: Matrix indexing completes and its journal entry is removed before webhook delivery, so a failed LLM batch has no durable retry source.

Git blame shows the 30-second timeout and current debounced LLM request flow were introduced together in commit `8a2d96f` (2026-06-14, “replace webhook HTTP POST with debounced LLM callback”). Commit `c1692e7` (2026-07-02) later added response-body logging, which made HTTP error responses visible but did not change timeout/retry behavior.

### Secondary failures

There are a few genuine upstream/edge HTTP errors, but they are rare and not the dominant recurring signature:

- 2026-07-15: `503 no available server`
- 2026-07-28: `404 page not found`
- 2026-07-30: one `403` HTML response, likely an ingress/security edge response

These should remain observable and retryable, but they do not explain the repeated `ReadTimeout` traces. The many intervening and subsequent `200 OK` calls also argue against treating them as the main root cause.

## Root cause

**Primary root cause:** the fixed 30-second HTTP read timeout is too short for the configured agentic `steven` workflow. Tool discovery/use, MCP calls, memory loading, and multiple Anthropic turns routinely push end-to-end latency beyond 30 seconds.

**Data-loss amplifier:** failed batches are discarded before delivery and never retried. A transient timeout or 5xx therefore becomes permanent loss of all messages in that batch.

Large batches can further increase prompt size and agent work. The 30-minute debounce is reset by every incoming message, so busy periods may produce unusually large batches; this is contributory but not required for the failure, since one-message batches also time out.

## Suggested fix

1. Make the webhook timeout configurable (for example, `WEBHOOK_TIMEOUT_SECONDS`) and set the read timeout to a value appropriate for an agent/tool loop, initially **180-300 seconds**. Prefer an explicit `httpx.Timeout` so connect/write/pool timeouts can remain short while only the read timeout is extended.
2. Do not lose batches on transient failure. Requeue or durably journal them and retry with bounded exponential backoff and jitter. Preserve ordering and use a request/idempotency key if the upstream supports one, because a timed-out request may still complete server-side and tool actions may already have occurred.
3. Bound batch size independently of the debounce period (and split very large batches) to cap prompt size and latency. The 99-message batch is a useful stress case.
4. Retry `ReadTimeout`, connection errors, `429`, and `5xx`; normally do not blindly retry persistent `4xx` configuration/auth errors. Log attempt number, elapsed time, batch identifier, and exception class.
5. Consider a streaming or asynchronous job interface if OpenWebUI offers one. Merely increasing the timeout is the fastest mitigation, while durable retry/idempotency is the correctness fix.

No source or deployment changes were made during this investigation.
