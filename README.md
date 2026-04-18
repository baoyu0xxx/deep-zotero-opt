# deep-zotero

`deep-zotero` is a standalone Zotero RAG MCP service. After installation, the only formal entrypoints are:

- `deep-zotero.exe`
- `deep-zotero-index.exe`

The default deployment shape is HTTP MCP:

```powershell
deep-zotero.exe --transport streamable-http --host 127.0.0.1 --port 8765 --path /mcp
```

Then configure your MCP client with a `url` such as `http://127.0.0.1:8765/mcp`.

## Main Path

1. Install the package into a virtualenv.
2. Copy `config.example.json` to your active `config.json`.
3. Run `deep-zotero-index.exe --rebuild` once.
4. Run `deep-zotero.exe --transport streamable-http`.
5. Point your MCP client at the HTTP `url`.

`stdio` is still supported:

```powershell
deep-zotero.exe --transport stdio
```

## Config

Config lookup priority is:

1. `--config <path>`
2. `DEEP_ZOTERO_CONFIG`
3. `<project-root>/config.json`
4. `~/.config/deep-zotero/config.json`

Relative paths are resolved against the project root, never against the current shell `cwd`.

Default runtime paths:

```json
{
  "chroma_db_path": "./.runtime-data/chroma_qwen",
  "runtime_data_dir": "./.runtime-data",
  "runtime_tmp_dir": "./.runtime-tmp",
  "scratch_tmp_dir": "./.tmp",
  "model_cache_dir": "./.runtime-data/hf-cache",
  "runtime_log_path": "./.runtime-data/logs/deep-zotero.log",
  "ocr_debug_dir": "./.runtime-data/ocr-debug"
}
```

At runtime, `health_check` and the CLI print both:

- the raw configured path value
- the resolved absolute path

## Example Config

```json
{
  "zotero_data_dir": "D:/zotero_backup",
  "chroma_db_path": "./.runtime-data/chroma_qwen",
  "runtime_data_dir": "./.runtime-data",
  "runtime_tmp_dir": "./.runtime-tmp",
  "scratch_tmp_dir": "./.tmp",
  "model_cache_dir": "./.runtime-data/hf-cache",
  "runtime_log_path": "./.runtime-data/logs/deep-zotero.log",
  "ocr_debug_dir": "./.runtime-data/ocr-debug",
  "embedding_provider": "qwen_inprocess",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "embedding_dimensions": 1024,
  "embedding_batch_size": 8,
  "embedding_device": null,
  "embedding_allow_download": false,
  "vision_enabled": false
}
```

## Indexing CLI

Incremental update:

```powershell
deep-zotero-index.exe --update
```

Full rebuild:

```powershell
deep-zotero-index.exe --rebuild
```

Single item:

```powershell
deep-zotero-index.exe --update --item-key ABC123
```

Title filter:

```powershell
deep-zotero-index.exe --update --title "family business"
```

OCR smoke test:

```powershell
deep-zotero-index.exe --ocr-smoke-test-only
```

Health check plus update:

```powershell
deep-zotero-index.exe --health-check --update
```

Health check only:

```powershell
deep-zotero-index.exe --health-check-only
```

Useful switches:

- `--ocr-mode auto|always|off`
- `--no-vision`
- `--embedding-provider qwen_inprocess|local|gemini`
- `--embedding-model ...`
- `--embedding-device ...`
- `--temp-dir ...`

## MCP Tools

Formal tools exposed by the MCP server:

- `health_check`
- `get_index_stats`
- `index_library`
- `search_diverse_papers`
- `search_topic`
- `search_papers`
- `get_passage_context`
- other existing retrieval helpers

`health_check` verifies:

- config load and validation
- raw/resolved path mapping
- Zotero DB presence
- Chroma path status
- embedding model readiness
- optional index stats

## Embedding Bootstrap

The default embedding path is in-process Qwen (`qwen_inprocess` provider).

`deep-zotero.exe` boot flow is:

1. Load config.
2. Resolve project-relative runtime paths.
3. Validate Zotero paths.
4. Ensure runtime directories exist.
5. Load and warm up the embedding model in-process.
6. Start MCP.

Failures are surfaced as explicit startup errors instead of a generic "tool unavailable".

## Notes

- Root-level scripts such as `full_rag_update.py` and `mcp_server_launcher.py` are compatibility wrappers only.
- Skill-based prompting can improve retrieval strategy, but skill setup is not required for MCP startup or search.
- `.runtime-data/`, `.runtime-tmp/`, and `.tmp/` are ignored by git and are intended for deploy-local runtime state.
