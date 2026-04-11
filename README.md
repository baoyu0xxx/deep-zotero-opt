# DeepZotero

Semantic search over a Zotero library. PDFs are extracted into text, tables, and figures, chunked, embedded, and stored in ChromaDB. An MCP server exposes the index to Claude Code or any other MCP client for semantic search, diversified paper retrieval, boolean search, table/figure search, context expansion, citation lookup, indexing, and cost inspection.

## Workspace Guidance

Use the repository's repo-local virtual environment and MCP settings for your machine. The server is started with the repo's Python entrypoint and can be pointed at any valid Zotero data directory and ChromaDB path through configuration.

## What It Extracts

- Text: section-aware chunks with overlap, classified by document section such as abstract, methods, results, and conclusion
- Tables: vision-based extraction via Claude Haiku 4.5 when enabled, with fallback heuristics when vision is disabled
- Figures: detected with captions, extracted as PNGs, searchable by caption text

## Requirements

- Python 3.10+
- A Zotero installation or backup containing `zotero.sqlite` and `storage/`
- One embedding backend:
  - Gemini API for `embedding_provider: "gemini"`
  - Local Chroma embedding fallback for `embedding_provider: "local"`
  - A local or remote OpenAI-compatible `/v1/embeddings` endpoint for `embedding_provider: "openai_compatible"`
- An Anthropic API key if you want vision-based table extraction

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

For vision table extraction:

```bash
.venv/Scripts/python.exe -m pip install -e ".[vision]"
```

## Setup

### 1. Configuration

```bash
mkdir -p ~/.config/deep-zotero
cp config.example.json ~/.config/deep-zotero/config.json
```

Minimal Gemini-based config:

```json
{
  "zotero_data_dir": "~/Zotero",
  "chroma_db_path": "~/.local/share/deep-zotero/chroma",
  "embedding_provider": "gemini",
  "embedding_model": "gemini-embedding-001",
  "embedding_dimensions": 768,
  "gemini_api_key": "YOUR_GEMINI_KEY",
  "anthropic_api_key": "YOUR_ANTHROPIC_KEY"
}
```

Example OpenAI-compatible config:

```json
{
  "zotero_data_dir": "/path/to/zotero",
  "chroma_db_path": "/path/to/chroma",
  "embedding_provider": "openai_compatible",
  "embedding_model": "your-embedding-model",
  "embedding_dimensions": 1024,
  "embedding_base_url": "http://localhost:8000/v1",
  "embedding_api_key": "YOUR_API_KEY",
  "embedding_query_instruction": null,
  "embedding_auto_start": true,
  "embedding_startup_timeout": 180.0,
  "vision_enabled": false,
  "ocr_language": "eng"
}
```

### 2. Embedding And API Keys

Gemini mode:

- Set `embedding_provider` to `"gemini"`
- Set `gemini_api_key` in config, or use `GEMINI_API_KEY`

Local fallback mode:

- Set `embedding_provider` to `"local"`
- No API key is required
- This uses Chroma's default `all-MiniLM-L6-v2` embedding model

OpenAI-compatible mode:

- Set `embedding_provider` to `"openai_compatible"`
- Set `embedding_base_url` to your embeddings endpoint
- Set `embedding_api_key` to the value required by that endpoint
- Set `embedding_model` and `embedding_dimensions` to match the served model
- Optional: set `embedding_query_instruction` to prepend a retrieval instruction to query embeddings

Anthropic vision mode:

- Set `anthropic_api_key` in config, or use `ANTHROPIC_API_KEY`
- If omitted, table extraction falls back to non-vision heuristics

To disable vision extraction entirely:

```json
{
  "vision_enabled": false
}
```

### 3. Index Your Library

```bash
deep-zotero-index -v
```

To test with a subset first:

```bash
deep-zotero-index --limit 10 -v
```

This reads the Zotero SQLite database in read-only mode, extracts text/tables/figures from each PDF, chunks the text, embeds via the configured backend, and stores everything in ChromaDB.

CLI options:

| Flag | Description |
|------|-------------|
| `--force` | Delete and rebuild index for all matching items |
| `--limit N` | Only index N items |
| `--item-key KEY` | Index a single Zotero item |
| `--title PATTERN` | Regex filter on title (case-insensitive) |
| `--no-vision` | Skip vision table extraction for this run |
| `--ocr-mode auto\|always\|off` | Control OCR for this run |
| `--temp-dir PATH` | Put Python/native temporary files under a specific directory |
| `--config PATH` | Use a different config file |
| `-v` | Debug logging |

The indexer is incremental. Use `--force` after changing chunking settings, embedding dimensions, OCR mode, or OCR language.

### 4. Register The MCP Server

Add this to your Claude Code settings:

```json
{
  "mcpServers": {
    "deep-zotero": {
      "command": "/path/to/repo/.venv/bin/python",
      "args": ["/path/to/repo/tools/mcp_server_launcher.py"]
    }
  }
}
```

Use the launcher instead of calling `deep_zotero.server` directly when you run
with `embedding_provider: "openai_compatible"` and `embedding_auto_start: true`.
The launcher probes the configured `/v1/embeddings` endpoint, starts the bundled
Qwen embedding server when needed, and then hands off to the MCP stdio server.
It also prefers a complete local Hugging Face snapshot and falls back to the
shared Hugging Face cache if the dedicated cache only contains partial downloads.

Restart the client after updating MCP config.

## Configuration Reference

### Zotero

| Field | Default | Description |
|---|---|---|
| `zotero_data_dir` | `~/Zotero` | Path to Zotero data directory containing `zotero.sqlite` and `storage/` |
| `chroma_db_path` | `~/.local/share/deep-zotero/chroma` | Path to on-disk ChromaDB index |

### Embedding

| Field | Default | Description |
|---|---|---|
| `embedding_provider` | `"gemini"` | `"gemini"`, `"local"`, or `"openai_compatible"` |
| `embedding_model` | `"gemini-embedding-001"` | Embedding model name |
| `embedding_dimensions` | `768` | Output vector dimensions |
| `gemini_api_key` | `null` | Falls back to `GEMINI_API_KEY` |
| `embedding_base_url` | `null` | Base URL for OpenAI-compatible embeddings |
| `embedding_api_key` | `null` | API key for OpenAI-compatible embeddings |
| `embedding_query_instruction` | `null` | Optional query prefix/instruction |
| `embedding_batch_size` | `8` | Batch size for OpenAI-compatible embedding requests |
| `embedding_auto_start` | `false` | Whether to auto-start the embedding endpoint |
| `embedding_startup_timeout` | `180.0` | Time to wait for embedding endpoint readiness |
| `embedding_start_command` | `null` | Optional startup command for embedding service |
| `embedding_timeout` | `120.0` | Per-request timeout |
| `embedding_max_retries` | `3` | Retry count for failed embedding calls |

### Chunking

| Field | Default | Description |
|---|---|---|
| `chunk_size` | `400` | Target chunk size in tokens |
| `chunk_overlap` | `100` | Overlap between adjacent chunks |

### Vision

| Field | Default | Description |
|---|---|---|
| `vision_enabled` | `true` | Enable vision-based table extraction |
| `vision_model` | `"claude-haiku-4-5-20251001"` | Anthropic model for table transcription |
| `anthropic_api_key` | `null` | Falls back to `ANTHROPIC_API_KEY` |

### Reranking

| Field | Default | Description |
|---|---|---|
| `rerank_enabled` | `true` | Enable composite score reranking |
| `rerank_alpha` | `0.7` | Similarity exponent |
| `rerank_section_weights` | `null` | Optional default section weight overrides |
| `rerank_journal_weights` | `null` | Optional default quartile weight overrides |
| `oversample_multiplier` | `3` | Oversample factor before reranking |
| `oversample_topic_factor` | `5` | Additional factor for topic-style searches |
| `stats_sample_limit` | `10000` | Max chunk sample size for stats |

### OCR

| Field | Default | Description |
|---|---|---|
| `ocr_language` | `"eng"` | OCR language code |

### OpenAlex

| Field | Default | Description |
|---|---|---|
| `openalex_email` | `null` | Optional OpenAlex polite-pool email |

## MCP Tools

### Semantic Search

`search_papers`

- Passage-level semantic search
- Best when you want the strongest raw passages
- Supports `required_terms` for exact whole-word filtering on top of semantic retrieval

`search_topic`

- Paper-level topic search deduplicated by document
- Good for ranking whole papers rather than passages

`search_diverse_papers`

- Diversified paper-level semantic search
- Returns distinct papers plus a few non-overlapping top passages per paper
- Recommended as the first tool for literature discovery because it prevents one paper from dominating the result set

`search_tables`

- Semantic search over table content

`search_figures`

- Semantic search over figure captions

### Boolean Search

`search_boolean`

- Exact word matching via Zotero's full-text index
- Useful when exact term presence matters more than semantic similarity

### Context Expansion

`get_passage_context`

- Expands context around a passage hit

### Citation Graph

`find_citing_papers`

- Papers citing the current document

`find_references`

- References cited by the current document

`get_citation_count`

- Citation and reference counts

### Index Management

`index_library`

- Trigger indexing from the MCP client

`get_index_stats`

- Inspect document and chunk counts plus section coverage

`get_reranking_config`

- Inspect valid reranking weights and effective defaults

`get_vision_costs`

- Inspect vision API batch usage and cost summaries

## Reranking

Search results are scored as:

```text
composite_score = similarity^alpha * section_weight * journal_weight
```

Default section weights:

| Section | Weight |
|---------|--------|
| results | 1.0 |
| conclusion | 1.0 |
| table | 0.9 |
| methods | 0.85 |
| abstract | 0.75 |
| background | 0.7 |
| unknown | 0.7 |
| discussion | 0.65 |
| introduction | 0.5 |
| preamble | 0.3 |
| appendix | 0.3 |
| references | 0.1 |

Default journal weights:

- Q1 = 1.0
- Q2 = 0.85
- Q3 = 0.65
- Q4 = 0.45
- unknown = configurable via overrides

## Shared Filter Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `author` | string | Case-insensitive substring match against author names |
| `tag` | string | Case-insensitive substring match against Zotero tags |
| `collection` | string | Case-insensitive substring match against collection names |
| `year_min` / `year_max` | int | Publication year range |
| `section_weights` | dict | Override section weights for a call |
| `journal_weights` | dict | Override journal quartile weights for a call |
| `required_terms` | list | Exact whole-word matches required in passage text |

## Debug Viewer

`tools/debug_viewer.py` is a PyQt6 browser for inspecting the ChromaDB index, including papers, tables, figures, and individual chunks.

```bash
.venv/Scripts/python.exe tools/debug_viewer.py
```
