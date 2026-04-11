from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

if hasattr(sys, "_base_executable"):
    sys._base_executable = sys.executable
multiprocessing.set_executable(sys.executable)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen_embedding_server.py",
        description="OpenAI-compatible local embedding server for Qwen embeddings.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto when omitted.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads even when a cached model exists.",
    )
    return parser.parse_args()


def _has_cached_model(cache_dir: str | None, model_name: str) -> bool:
    return _find_complete_snapshot(cache_dir, model_name) is not None


def _snapshot_has_weights(snapshot_dir: Path) -> bool:
    weight_names = ("model.safetensors", "pytorch_model.bin")
    return any((snapshot_dir / name).exists() for name in weight_names)


def _model_cache_root(cache_dir: str | None, model_name: str) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / f"models--{model_name.replace('/', '--')}"


def _find_complete_snapshot(cache_dir: str | None, model_name: str) -> Path | None:
    model_cache = _model_cache_root(cache_dir, model_name)
    if model_cache is None:
        return None
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir() and _snapshot_has_weights(snapshot):
            return snapshot
    return None


def _default_hf_cache_dir() -> Path:
    if os.name == "nt":
        return Path.home() / ".cache" / "huggingface" / "hub"
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _resolve_model_source(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Resolve a usable local snapshot before falling back to model id download."""
    model_kwargs: dict[str, Any] = {}

    dedicated_snapshot = _find_complete_snapshot(args.cache_dir, args.model)
    if dedicated_snapshot is not None:
        logger.info("Using dedicated cached snapshot: %s", dedicated_snapshot)
        if args.cache_dir:
            os.environ.setdefault("HF_HOME", args.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
        if not args.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        return str(dedicated_snapshot), model_kwargs

    shared_snapshot = _find_complete_snapshot(str(_default_hf_cache_dir()), args.model)
    if shared_snapshot is not None:
        logger.info("Dedicated cache incomplete; using shared Hugging Face snapshot: %s", shared_snapshot)
        return str(shared_snapshot), model_kwargs

    if args.cache_dir:
        model_kwargs["cache_folder"] = args.cache_dir
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
    return args.model, model_kwargs


def _create_app(args: argparse.Namespace) -> FastAPI:
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model_source, model_kwargs = _resolve_model_source(args)
    if args.device:
        model_kwargs["device"] = args.device
    if (
        model_source == args.model
        and _has_cached_model(args.cache_dir, args.model)
        and not args.allow_download
    ):
        model_kwargs["local_files_only"] = True
        os.environ["HF_HUB_OFFLINE"] = "1"

    model = SentenceTransformer(model_source, **model_kwargs)
    app = FastAPI(title="Qwen embedding server")

    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_with_backoff(texts: list[str]) -> tuple[Any, int]:
        current_batch_size = max(1, args.batch_size)
        last_error: BaseException | None = None

        while current_batch_size >= 1:
            try:
                vectors = model.encode(
                    texts,
                    batch_size=current_batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return vectors, current_batch_size
            except torch.OutOfMemoryError as exc:
                last_error = exc
                logger.warning(
                    "Embedding request hit CUDA OOM for %d texts at batch_size=%d; retrying smaller batch",
                    len(texts),
                    current_batch_size,
                )
                _clear_cuda_cache()
                if current_batch_size == 1:
                    break
                current_batch_size = max(1, current_batch_size // 2)

        assert last_error is not None
        raise last_error

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        texts = request.input if isinstance(request.input, list) else [request.input]
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")
        if not all(isinstance(text, str) for text in texts):
            raise HTTPException(status_code=400, detail="input must be a string or string array")

        try:
            vectors, used_batch_size = _encode_with_backoff(texts)
        except torch.OutOfMemoryError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "embedding request exhausted CUDA memory",
                    "error_type": "cuda_oom",
                    "input_count": len(texts),
                    "requested_batch_size": args.batch_size,
                    "final_batch_size": 1,
                    "model": args.model,
                    "device": args.device or "auto",
                    "retry_hint": "reduce client embedding batch size or shorten texts",
                    "error": str(exc),
                },
            ) from exc

        now = int(time.time())
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector.astype(float).tolist(),
            }
            for index, vector in enumerate(vectors)
        ]
        return {
            "id": f"embd-{uuid.uuid4().hex}",
            "object": "list",
            "created": now,
            "model": request.model or args.model,
            "data": data,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
            "debug": {
                "batch_size": used_batch_size,
                "input_count": len(texts),
            },
        }

    return app


def main() -> None:
    args = _parse_args()
    app = _create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
