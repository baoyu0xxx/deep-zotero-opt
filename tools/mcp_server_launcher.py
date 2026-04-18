from __future__ import annotations

import sys
from pathlib import Path


def _load_server_main():
    try:
        from deep_zotero.server import main as server_main
        return server_main
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        src_dir = repo_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from deep_zotero.server import main as server_main
        return server_main


def main() -> int:
    print("Deprecated helper: use deep-zotero.exe directly.", file=sys.stderr)
    server_main = _load_server_main()
    return server_main()


if __name__ == "__main__":
    raise SystemExit(main())
