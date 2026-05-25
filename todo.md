# Code Review TODO

## Medium Severity

- [x] **2.4 — Module-level mutable globals** (`server.py:27-28`)
  `_matrix_client` and `_webhook_dispatcher` moved to `app.state`; all setup/teardown now lives in `lifespan()`.

- [x] **2.5 — Lazy HTTP client initialization** (`webhook.py:33-36`)
  Removed `_get_http()`; added `async start()` that initialises `AsyncClient` once; `_http_post` reuses it without closing.

- [x] **3.1 — Complex, deeply nested filter builder** (`vector_store.py:128-234`)
  Extracted `_room_condition`, `_exact_sender_condition`, `_timestamp_condition`, `_fuzzy_sender_min_should` as module-level helpers; `_build_filter` now delegates to them.

## Low Severity

- [x] **1.1 — Duplicated `SearchResult` construction** (`vector_store.py:237-252, 334-347`)
  `_results_from_hits` uses `getattr(hit, "score", default_score)`; `_scroll_once` now delegates to it.

- [x] **1.3 — Repeated `TextContent(json.dumps(...))` boilerplate** (`server.py:127, 164, 173, 182, 188`)
  Extracted `_json_response()` helper; all five call sites updated.

- [x] **3.2 — Nested try/except in `_on_message`** (`matrix_client.py:451-471`)
  Extracted `_save_pending_index_safe() -> bool`; `_on_message` now has two nesting levels instead of three.

- [x] **4.2 — Duplicate import of `RoomMessageText`** (`matrix_client.py:13, 18`)
  Removed aliased import; all `RoomMessageTextEvent` usages replaced with `RoomMessageText`.

- [x] **5.5 — File-not-found logged at wrong level** (`matrix_client.py:285-288, 311-314`)
  Changed `pass` to `logger.debug(...)` in both `_load_buffer` and `_load_pending_index`.

- [x] **6.2 — No cross-field config validation** (`config.py`)
  Added `@field_validator` for positive-integer fields, non-negative `backfill_pages_max`, and port-range validation.
