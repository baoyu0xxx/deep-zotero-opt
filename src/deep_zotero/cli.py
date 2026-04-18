"""Unified indexing CLI for deep-zotero."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import Config
from .indexer import Indexer
from .mcp_bootstrap import collect_health_status


SMOKE_TEST_FIXTURE = "tests/fixtures/papers/noname3.pdf"


def _configure_runtime_env(config: Config, temp_dir: Path | None = None) -> None:
    root = temp_dir if temp_dir is not None else config.runtime_tmp_dir
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = config.runtime_data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(root)
    os.environ["HF_HOME"] = str(config.model_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(config.model_cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ["PADDLE_HOME"] = str(config.runtime_data_dir / "paddle")
    os.environ["PYTHONPYCACHEPREFIX"] = str(config.scratch_tmp_dir / "pycache")


def _apply_embedding_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.embedding_provider != "config":
        config.embedding_provider = args.embedding_provider
    if args.embedding_provider == "local":
        config.embedding_model = "all-MiniLM-L6-v2"
        config.embedding_dimensions = 384
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.embedding_dimensions:
        config.embedding_dimensions = args.embedding_dimensions
    if args.embedding_device:
        config.embedding_device = args.embedding_device
    if args.embedding_allow_download:
        config.embedding_allow_download = True


def _run_ocr_smoke_test(config: Config) -> dict:
    import onnxruntime as ort
    import pymupdf

    from .pdf_processor import extract_document

    fixture_pdf = config.project_root / SMOKE_TEST_FIXTURE
    if not fixture_pdf.exists():
        raise FileNotFoundError(f"OCR smoke-test fixture not found: {fixture_pdf}")

    smoke_dir = config.runtime_tmp_dir / "ocr-smoke-test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    sample_pdf = smoke_dir / "noname3-page1-scanned.pdf"

    src = pymupdf.open(str(fixture_pdf))
    try:
        page = src[0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        out = pymupdf.open()
        try:
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
            out.save(str(sample_pdf))
        finally:
            out.close()
    finally:
        src.close()

    start = time.perf_counter()
    extraction = extract_document(
        sample_pdf,
        ocr_mode="always",
        ocr_language=config.ocr_language,
        vision_api=None,
    )
    elapsed = time.perf_counter() - start

    if extraction.stats.get("ocr_pages", 0) < 1:
        raise RuntimeError("OCR smoke test did not OCR any page")
    if len(extraction.full_markdown.strip()) < 100:
        raise RuntimeError("OCR smoke test produced too little text")

    return {
        "sample_pdf": str(sample_pdf),
        "elapsed_sec": round(elapsed, 2),
        "providers": ort.get_available_providers(),
        "pages": len(extraction.pages),
        "markdown_chars": len(extraction.full_markdown),
        "quality_grade": extraction.quality_grade,
        "stats": extraction.stats,
    }


def _serialize_summary(config: Config, result: dict, mode: str, ocr_mode: str) -> dict:
    return {
        "project_root": str(config.project_root),
        "config_path": str(config.config_path) if config.config_path else None,
        "mode": mode,
        "ocr_mode": ocr_mode,
        "paths": config.path_report(),
        "indexed": result["indexed"],
        "already_indexed": result["already_indexed"],
        "skipped": result["skipped"],
        "failed": result["failed"],
        "empty": result["empty"],
        "quality_distribution": result.get("quality_distribution"),
        "extraction_stats": result.get("extraction_stats"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deep-zotero-index",
        description="Unified CLI for full rebuilds, incremental updates, and OCR smoke checks.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--rebuild", action="store_true", help="Delete and rebuild matching index entries.")
    mode_group.add_argument("--update", action="store_true", help="Incrementally index new or changed items (default).")
    parser.add_argument("--force", action="store_true", help="Backward-compatible alias for --rebuild.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of matching items to index.")
    parser.add_argument("--item-key", type=str, default=None, help="Index only this Zotero item key.")
    parser.add_argument("--title", type=str, default=None, help="Regex pattern to filter items by title.")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision-based table extraction for this run.")
    parser.add_argument("--ocr-mode", choices=("auto", "always", "off"), default="auto")
    parser.add_argument("--temp-dir", type=str, default=None, help="Override runtime temp dir for this run.")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file.")
    parser.add_argument(
        "--embedding-provider",
        choices=("config", "qwen_inprocess", "local", "gemini"),
        default="config",
        help="Override embedding provider for this run.",
    )
    parser.add_argument("--embedding-model", type=str, default=None, help="Override embedding model.")
    parser.add_argument("--embedding-dimensions", type=int, default=None, help="Override embedding dimensions.")
    parser.add_argument("--embedding-device", type=str, default=None, help="Force embedding device (e.g., cpu/cuda).")
    parser.add_argument(
        "--embedding-allow-download",
        action="store_true",
        help="Allow model download when no local snapshot is available.",
    )
    parser.add_argument("--health-check", action="store_true", help="Print health diagnostics before indexing.")
    parser.add_argument("--health-check-only", action="store_true", help="Run diagnostics and exit.")
    parser.add_argument("--ocr-smoke-test", action="store_true", help="Run the built-in OCR smoke test before indexing.")
    parser.add_argument("--ocr-smoke-test-only", action="store_true", help="Run only the OCR smoke test.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )

    config = Config.load(args.config)
    config.ensure_runtime_dirs()
    _apply_embedding_overrides(config, args)
    if args.no_vision:
        config.vision_enabled = False

    temp_dir = Path(args.temp_dir).expanduser() if args.temp_dir else config.runtime_tmp_dir
    _configure_runtime_env(config, temp_dir=temp_dir)

    if args.health_check:
        print(json.dumps(collect_health_status(config), ensure_ascii=False, indent=2))
    if args.health_check_only:
        print(json.dumps(collect_health_status(config), ensure_ascii=False, indent=2))
        return 0

    smoke_result = None
    if args.ocr_smoke_test or args.ocr_smoke_test_only:
        smoke_result = _run_ocr_smoke_test(config)
        print(json.dumps({"ocr_smoke_test": smoke_result}, ensure_ascii=False, indent=2))
        if args.ocr_smoke_test_only:
            return 0

    errors = config.validate()
    if errors:
        for error in errors:
            print(f"Config error: {error}", file=sys.stderr)
        return 1

    mode = "rebuild" if args.rebuild or args.force else "update"

    indexer = Indexer(config)
    result = indexer.index_all(
        force_reindex=(mode == "rebuild"),
        limit=args.limit,
        item_key=args.item_key,
        title_pattern=args.title,
        ocr_mode=args.ocr_mode,
    )

    summary = _serialize_summary(config, result, mode, args.ocr_mode)
    if smoke_result is not None:
        summary["ocr_smoke_test"] = smoke_result

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures = [r for r in result["results"] if r.status == "failed"]
    if failures:
        print(
            json.dumps(
                {
                    "failed_items": [
                        {"item_key": item.item_key, "title": item.title, "reason": item.reason}
                        for item in failures
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 1 if result["failed"] > 0 and result["indexed"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
