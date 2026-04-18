from __future__ import annotations

import json
import os
from pathlib import Path

from deep_zotero.config import Config, DEFAULT_CHROMA_DB_PATH, discover_project_root


def test_default_chroma_path_is_project_relative(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DEEP_ZOTERO_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path)

    config = Config.load()
    project_root = discover_project_root()

    assert config.raw_path_values["chroma_db_path"] == DEFAULT_CHROMA_DB_PATH
    assert config.chroma_db_path == (project_root / DEFAULT_CHROMA_DB_PATH).resolve()


def test_relative_paths_ignore_cwd_and_resolve_from_project_root(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "custom-config.json"
    payload = {
        "zotero_data_dir": "./fixtures/zotero",
        "chroma_db_path": "./.runtime-data/custom-chroma",
        "runtime_tmp_dir": "./.runtime-tmp/custom",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("DEEP_ZOTERO_CONFIG", str(config_path))

    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    config = Config.load()
    project_root = discover_project_root()

    assert config.zotero_data_dir == (project_root / "fixtures/zotero").resolve()
    assert config.chroma_db_path == (project_root / ".runtime-data/custom-chroma").resolve()
    assert config.runtime_tmp_dir == (project_root / ".runtime-tmp/custom").resolve()


def test_explicit_absolute_path_stays_absolute(monkeypatch, tmp_path: Path):
    absolute_chroma = (tmp_path / "absolute-chroma").resolve()
    config_path = tmp_path / "absolute-config.json"
    config_path.write_text(
        json.dumps({"chroma_db_path": str(absolute_chroma)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEP_ZOTERO_CONFIG", str(config_path))

    config = Config.load()

    assert config.chroma_db_path == absolute_chroma
    assert os.path.isabs(config.raw_path_values["chroma_db_path"])


def test_removed_sidecar_keys_report_migration_error(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "legacy-config.json"
    config_path.write_text(
        json.dumps(
            {
                "embedding_provider": "qwen_inprocess",
                "embedding_base_url": "http://127.0.0.1:8000/v1",
                "embedding_api_key": "EMPTY",
                "embedding_auto_start": True,
                "embedding_start_command": "python -m deep_zotero.qwen_embedding_server",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEP_ZOTERO_CONFIG", str(config_path))

    config = Config.load()
    errors = config.validate()

    assert any("Deprecated sidecar embedding config keys detected" in e for e in errors)
