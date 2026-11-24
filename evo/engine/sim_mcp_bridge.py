"""Minimal stdio MCP server bridging a CLI agent (Codex) to the sim worker.

Speaks MCP's stdio transport: newline-delimited JSON-RPC 2.0 (one JSON
object per line, no Content-Length framing). It exposes a single tool,
``execute_python_code``, and forwards each call to the persistent sim
worker on the TCP port named by the ``EVO_SIM_PORT`` env var (see
``evo.engine.sim_manager.send_to_sim_worker``).

Pure stdlib on the protocol side so `codex exec` can launch it with just
``python -m evo.engine.sim_mcp_bridge`` — no external MCP package and no
``~/.codex/config.toml`` prerequisite (the loop passes the full server
definition via ``-c`` overrides; see ``codex_backend._sim_mcp_config_args``).
"""
from __future__ import annotations

import json
import os
import sys

from evo.engine.sim_manager import send_to_sim_worker

PROTOCOL_VERSION_FALLBACK = "2024-11-05"

TOOL_DEFINITION = {
    "name": "execute_python_code",
    "description": (
        "Execute Python code in a persistent Python session with the frozen "
        "`assembly` SDK preloaded and the immutable runtime globals "
        "(ASSEMBLY_SDK_DIR, SKILL_DIR, INPUT_DIR, OUTPUT_DIR) injected. "
        "Variables, imports, and helper functions persist across calls. "
        "Returns captured stdout/stderr plus the value of a trailing "
        "expression or a variable named `result`."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _text_result(text: str, *, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def _call_tool(params: dict) -> dict:
    name = str(params.get("name", ""))
    if name != "execute_python_code":
        return _text_result(f"unknown tool: {name}", is_error=True)
    code = str((params.get("arguments") or {}).get("code", ""))
    port = int(os.environ.get("EVO_SIM_PORT", "9876"))
    try:
        resp = send_to_sim_worker(code, port)
    except Exception as exc:  # noqa: BLE001 - surface transport failures as text
        from evo.engine.sim_manager import memory_kill_notice
        return _text_result(memory_kill_notice(port) or f"socket error: {exc}", is_error=True)
    parts: list[str] = []
    if resp.get("stdout"):
        parts.append(str(resp["stdout"]).rstrip())
    if resp.get("stderr"):
        parts.append("STDERR:\n" + str(resp["stderr"]).rstrip())
    is_error = resp.get("status") != "ok"
    if is_error:
        parts.append(f"SIM ERROR: {resp.get('message', '?')}")
    elif resp.get("result") not in ({}, None, ""):
        parts.append("RESULT: " + json.dumps(resp["result"], default=str))
    return _text_result("\n".join(parts) or "(no output)", is_error=is_error)


def _handle(request: dict) -> dict | None:
    """Return the JSON-RPC result for one request, or None for notifications."""
    method = str(request.get("method", ""))
    params = request.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": (
                params.get("protocolVersion") or PROTOCOL_VERSION_FALLBACK
            ),
            "serverInfo": {"name": "sim", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        }
    if method == "tools/list":
        return {"tools": [TOOL_DEFINITION]}
    if method == "tools/call":
        return _call_tool(params)
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            })
            continue
        request_id = request.get("id")
        method = str(request.get("method", ""))
        if method.startswith("notifications/"):
            continue  # e.g. notifications/initialized — nothing to answer
        result = _handle(request)
        if request_id is None:
            continue  # notification-style request; no reply expected
        if result is None:
            _write({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })
        else:
            _write({"jsonrpc": "2.0", "id": request_id, "result": result})


if __name__ == "__main__":
    main()
