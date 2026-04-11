# DeepZotero

Semantic search over a Zotero library. PDFs are extracted (text, tables, figures), chunked, embedded, and stored in ChromaDB. An MCP server exposes the index to Claude Code (or any MCP client) as 14 tools for semantic search, boolean search, table/figure search, context expansion, citation graph lookup, indexing, and cost tracking.

## Workspace Guidance

- Open Codex at `D:\pyproject`. That is the trusted workspace root.
- `zotero_rag&mcp` is a helper/wrapper workspace only.
- `deep-zotero` remains the canonical MCP entrypoint and is the repo referenced by the server config.

## What it extracts

- **Text** — section-aware chunks with overlap, classified by document section (abstract, methods, results, etc.)
- **Tables** — vision-based extraction via Claude Haiku 4.5. Each table is rendered to PNG and transcribed to structured markdown (headers, rows, footnotes). Falls back to PyMuPDF heuristics if vision is disabled.
- **Figures** — detected with captions, extracted as PNGs, searchable by caption text.

## Requirements

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/app/apikey) for embeddings (unless using `embedding_provider: "local"`)
- An [Anthropic API key](https://console.anthropic.com/) for vision-based table extraction (optional but recommended)
- A Zotero installation with PDFs in `storage/`

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

Edit `~/.config/deep-zotero/config.json`:

```json
{
    "zotero_data_dir": "~/Zotero",
    "chroma_db_path": "~/.local/share/deep-zotero/chroma",
    "gemini_api_key": "YOUR_GEMINI_KEY",
    "anthropic_api_key": "YOUR_ANTHROPIC_KEY"
}
```

All other fields have sensible defaults. You can also set `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` as environment variables instead.

### 2. API keys

**Gemini (required for default embeddings):**
Get a key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). Set it as `gemini_api_key` in config or `GEMINI_API_KEY` env var. If you don't want to use Gemini, set `"embedding_provider": "local"` to use ChromaDB's built-in all-MiniLM-L6-v2 model (no API key needed, lower quality).

**Anthropic (required for vision table extraction):**
Get a key at [console.anthropic.com](https://console.anthropic.com/). Set it as `anthropic_api_key` in config or `ANTHROPIC_API_KEY` env var. Without this key, tables are still extracted via PyMuPDF heuristics but accuracy on complex tables is lower. Vision extraction uses the Anthropic Batch API with Claude Haiku 4.5 — cost is roughly $0.016 per table, with prompt caching reducing cost on large batches.

To disable vision extraction entirely:

```json
{
    "vision_enabled": false
}
```

### 3. Index your library

```bash
deep-zotero-index -v
```

To test with a subset first:

```bash
deep-zotero-index --limit 10 -v
```

This reads the Zotero SQLite database (read-only, safe while Zotero is open), extracts text/tables/figures from each PDF, chunks the text, embeds via Gemini, and stores everything in ChromaDB.

CLI options:

| Flag | Description |
|------|-------------|
| `--force` | Delete and rebuild index for all matching items |
| `--limit N` | Only index N items |
| `--item-key KEY` | Index a single Zotero item |
| `--title PATTERN` | Regex filter on title (case-insensitive) |
| `--no-vision` | Skip vision table extraction for this run |
| `--ocr-mode auto\|always\|off` | Control OCR for this run; `always` requires GPU OCR to be available |
| `--temp-dir PATH` | Put Python/native temporary files under a specific directory |
| `--config PATH` | Use a different config file |
| `-v` | Debug logging |

The indexer is incremental — it only processes items not already in the index. Use `--force` after changing `chunk_size`, `embedding_dimensions`, `ocr_language`, or OCR mode.

You can also trigger indexing from the MCP client via the `index_library` tool.

### 4. Register the MCP server

Add to your Claude Code settings (`~/.claude/settings.json`):

```json
{
    "mcpServers": {
        "deep-zotero": {
            "command": "/path/to/.venv/bin/python",
            "args": ["-m", "deep_zotero.server"]
        }
    }
}
```

On Windows:

```json
{
    "mcpServers": {
        "deep-zotero": {
            "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
            "args": ["-m", "deep_zotero.server"]
        }
    }
}
```

Restart Claude Code. All 14 tools will be available.

---

## Configuration reference

### Zotero

| Field | Default | Description |
|---|---|---|
| `zotero_data_dir` | `~/Zotero` | Path to Zotero's data directory (contains `zotero.sqlite` and `storage/`) |
| `chroma_db_path` | `~/.local/share/deep-zotero/chroma` | Where the ChromaDB index is stored on disk |

### Embedding

| Field | Default | Description |
|---|---|---|
| `embedding_provider` | `"gemini"` | `"gemini"` for Gemini API, `"local"` for ChromaDB's built-in all-MiniLM-L6-v2 (no key needed) |
| `embedding_model` | `"gemini-embedding-001"` | Gemini model name (only used when provider is `"gemini"`) |
| `embedding_dimensions` | `768` | Output vector dimensions. `gemini-embedding-001` supports 64-3072. Changing requires `--force` re-index |
| `gemini_api_key` | `null` | Falls back to `GEMINI_API_KEY` env var |
| `embedding_timeout` | `120.0` | Timeout in seconds for embedding API calls |
| `embedding_max_retries` | `3` | Max retries for failed embedding calls |

### Chunking

| Field | Default | Description |
|---|---|---|
| `chunk_size` | `400` | Target chunk size in tokens (~4 chars/token). Changing requires `--force` re-index |
| `chunk_overlap` | `100` | Overlap between consecutive chunks in tokens |

### Vision

| Field | Default | Description |
|---|---|---|
| `vision_enabled` | `true` | Enable vision table extraction during indexing |
| `vision_model` | `"claude-haiku-4-5-20251001"` | Anthropic model for table transcription |
| `anthropic_api_key` | `null` | Falls back to `ANTHROPIC_API_KEY` env var |

### Reranking

| Field | Default | Description |
|---|---|---|
| `rerank_enabled` | `true` | Enable composite score reranking |
| `rerank_alpha` | `0.7` | Similarity exponent (0-1). Lower = more metadata influence |
| `rerank_section_weights` | `null` | Override default section weights |
| `rerank_journal_weights` | `null` | Override default journal quartile weights |
| `oversample_multiplier` | `3` | Oversample factor before reranking |
| `oversample_topic_factor` | `5` | Additional factor for `search_topic` |
| `stats_sample_limit` | `10000` | Max chunks sampled for `get_index_stats` |

### OCR

| Field | Default | Description |
|---|---|---|
| `ocr_language` | `"eng"` | Tesseract language code for scanned pages (`"fra"`, `"deu"`, etc.). Changing requires `--force` re-index |

### OpenAlex

| Field | Default | Description |
|---|---|---|
| `openalex_email` | `null` | Email for OpenAlex polite pool (10 req/s vs 1 req/s). Falls back to `OPENALEX_EMAIL` env var |

---

## MCP tools

### Semantic search

**`search_papers`** — Passage-level semantic search. Returns matching text with surrounding context, reranked by composite score (similarity × section weight × journal weight). Supports `required_terms` for combining semantic search with exact word matching — each term must appear as a whole word in the passage.

Parameters: `query`, `top_k` (1-50), `context_chunks` (0-3), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types` (text/figure/table), `section_weights`, `journal_weights`, `required_terms` (list of words that must appear in passage).

**`search_topic`** — Paper-level topic search, deduplicated by document. Groups chunks by paper, scores by average and best composite relevance.

Parameters: `query`, `num_papers` (1-50), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types`, `section_weights`, `journal_weights`.

**`search_tables`** — Semantic search over table content (headers, cells, captions). Returns tables as markdown.

Parameters: `query`, `top_k` (1-30), `year_min`, `year_max`, `author`, `tag`, `collection`, `journal_weights`.

**`search_figures`** — Semantic search over figure captions. Returns figure metadata and paths to extracted PNGs.

Parameters: `query`, `top_k` (1-30), `year_min`, `year_max`, `author`, `tag`, `collection`.

**`search_diverse_papers`** - Diversified paper-level semantic search. Returns distinct papers plus up to a few non-overlapping top passages per paper, so one dominant document does not crowd out the rest of the result set.

Parameters: `query`, `num_papers` (1-50), `passages_per_paper` (1-5), `context_chunks` (0-3), `year_min`, `year_max`, `author`, `tag`, `collection`, `chunk_types`, `section_weights`, `journal_weights`, `required_terms`.

### Boolean search

**`search_boolean`** — Exact word matching via Zotero's native full-text index. Returns papers (not passages) matching AND/OR word queries. No phrase search, no stemming.

Parameters: `query` (space-separated terms), `operator` (AND/OR), `year_min`, `year_max`.

### Context expansion

**`get_passage_context`** — Expand context around a passage from `search_papers`. For table results, pass `table_page` and `table_index` to find body text citing the table.

Parameters: `doc_id`, `chunk_index`, `window` (1-5), `table_page`, `table_index`.

### Citation graph (OpenAlex)

Requires the document to have a DOI in Zotero.

**`find_citing_papers`** — Papers that cite a given document. Parameters: `doc_id`, `limit` (1-100).

**`find_references`** — Papers a document cites. Parameters: `doc_id`, `limit` (1-100).

**`get_citation_count`** — Citation and reference counts. Parameters: `doc_id`.

### Index management

**`index_library`** — Trigger indexing from the MCP client. Parameters: `force_reindex`, `limit`, `item_key`, `title_pattern`, `no_vision`, `ocr_mode`.

**`get_index_stats`** — Document/chunk/table/figure counts, section coverage, journal coverage.

**`get_reranking_config`** — Current reranking weights and valid override values.

**`get_vision_costs`** — Vision API batch usage and cost summary. Parameters: `last_n` (recent entries to show).

---

## Reranking

Search results are scored:

```
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

Default journal weights: Q1=1.0, Q2=0.85, Q3=0.65, Q4=0.45.

Override per-call via `section_weights` and `journal_weights` parameters. Set a section to 0 to exclude it. Disable reranking entirely with `"rerank_enabled": false`.

---

## Shared filter parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `author` | string | Case-insensitive substring match against author names |
| `tag` | string | Case-insensitive substring match against Zotero tags |
| `collection` | string | Case-insensitive substring match against collection names |
| `year_min` / `year_max` | int | Publication year range |
| `section_weights` | dict | Override section weights for this call |
| `journal_weights` | dict | Override journal quartile weights |
| `required_terms` | list | Exact whole-word matches required in passage (`search_papers` only) |

---

## Debug viewer

`tools/debug_viewer.py` is a PyQt6 browser for inspecting the ChromaDB index — view papers, tables (rendered markdown vs PDF), figures, and individual chunks.

```bash
.venv/Scripts/python.exe tools/debug_viewer.py
```
