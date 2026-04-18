from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anyio
import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_SERVER = ROOT / ".venv" / "Scripts" / "deep-zotero.exe"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mcp_smoke_test.py",
        description="Validate the deep-zotero MCP server over real MCP transport.",
    )
    parser.add_argument("--transport", choices=("stdio", "http"), default="http")
    parser.add_argument("--query", default="corporate governance")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--passages-per-paper", type=int, default=2)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--module", default="deep_zotero.server")
    parser.add_argument("--server-url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--http-timeout", type=float, default=90.0)
    parser.add_argument("--dump-tools", action="store_true")
    return parser.parse_args()


def _extract_results(payload: object) -> list[dict] | None:
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    return results if isinstance(results, list) else None


def _format_tool_schemas(tools: list) -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
        }
        for tool in tools
    ]


async def _run_stdio(args: argparse.Namespace) -> dict:
    if args.server.exists():
        params = StdioServerParameters(command=str(args.server), args=[])
    else:
        params = StdioServerParameters(
            command=str(args.python),
            args=["-m", args.module],
        )

    async with stdio_client(params) as (read, write):
        return await _run_session(args, read, write, "stdio")


async def _run_http(args: argparse.Namespace) -> dict:
    timeout = httpx.Timeout(args.http_timeout, read=args.http_timeout)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        async with streamable_http_client(args.server_url, http_client=http_client) as (read, write, _):
            return await _run_session(args, read, write, "http")


async def _run_session(args: argparse.Namespace, read, write, transport: str) -> dict:
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        stats = await session.call_tool("get_index_stats", {})
        if getattr(stats, "isError", False):
            raise RuntimeError(f"get_index_stats failed: {stats.content}")

        search = await session.call_tool(
            "search_diverse_papers",
            {
                "query": args.query,
                "top_k": args.top_k,
                "passages_per_paper": args.passages_per_paper,
                "context_window": args.context_window,
            },
        )
        if getattr(search, "isError", False):
            raise RuntimeError(f"search_diverse_papers failed: {search.content}")

    search_payload = search.structuredContent or {}
    papers = _extract_results(search_payload)
    payload = {
        "transport": transport,
        "initialized": True,
        "tools": [tool.name for tool in tools.tools],
        "index_stats": stats.structuredContent,
        "search_summary": {
            "query": args.query,
            "paper_count": len(papers) if isinstance(papers, list) else None,
            "first_doc_id": papers[0].get("doc_id") if isinstance(papers, list) and papers else None,
            "first_title": papers[0].get("doc_title") if isinstance(papers, list) and papers else None,
        },
    }
    if args.dump_tools:
        payload["tool_schemas"] = _format_tool_schemas(tools.tools)
    return payload


async def _run_smoke_test(args: argparse.Namespace) -> dict:
    if args.transport == "http":
        return await _run_http(args)
    return await _run_stdio(args)


def main() -> int:
    args = _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    payload = anyio.run(_run_smoke_test, args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
