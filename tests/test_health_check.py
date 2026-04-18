from __future__ import annotations

import deep_zotero.server as server


def test_health_check_reports_raw_and_resolved_paths(monkeypatch, mock_config):
    monkeypatch.setattr(server, "_config", mock_config)
    monkeypatch.setattr(
        server,
        "collect_health_status",
        lambda config: {
            "paths": config.path_report(),
            "validation_errors": [],
            "checks": {"embedding_model_ready": True},
        },
    )
    monkeypatch.setattr(
        server,
        "get_index_stats",
        lambda: {"total_documents": 3, "total_chunks": 9, "avg_chunks_per_doc": 3.0},
    )

    payload = server.health_check()

    assert payload["server"]["transport_recommendation"] == "streamable-http"
    assert "chroma_db_path" in payload["paths"]
    assert payload["paths"]["chroma_db_path"]["resolved"] == str(mock_config.chroma_db_path)
    assert payload["index_stats"]["total_documents"] == 3
