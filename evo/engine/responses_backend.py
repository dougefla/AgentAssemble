"""OpenAI Agents SDK backend using the Responses API exclusively.

The runtime supports OpenRouter and compatible OpenAI Responses providers.
The runner carries typed response items forward explicitly on every turn
instead of relying on provider-side conversation state.
"""
from __future__ import annotations

import asyncio
import ast
import base64
import dataclasses
import datetime as dt
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agents import (
    Agent as OpenAIAgent,
    ModelSettings,
    RunConfig,
    Runner,
    ToolOutputImage,
    ToolOutputText,
    function_tool,
)
from agents.mcp import MCPServer
from agents.models.openai_responses import OpenAIResponsesModel
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, TextContent, Tool
from openai import NOT_GIVEN, AsyncOpenAI

from evo.engine.model_defaults import DEFAULT_AGENT_EFFORT
from evo.engine import pricing
from evo.engine.sim_manager import send_to_sim_worker


DEFAULT_RESPONSES_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SERVICE_TIER = "default"
VALIDATOR_MAX_TURNS = 20
MAX_TEXT_TOOL_OUTPUT_CHARS = 24_000
MAX_GREP_MATCHES = 200
MAX_GREP_LINE_CHARS = 1_000


def _inject_runtime_globals(code: str, runtime_globals: dict[str, str]) -> str:
    """Inject authoritative runtime globals/env and ignore model overwrites.

    Defense in depth: the sim worker already re-injects these values before
    every execution, but merely trusting that is insufficient here because
    generated code can assign ``SKILL_DIR`` again before using it within one
    call. Rewrite every store/delete of a runtime-owned name to a harmless
    private name, while leaving reads bound to the injected value. Mirror the
    same values into ``os.environ`` because model-generated code sometimes
    uses that access path. This covers ordinary assignments, loop targets,
    unpacking, augmented assignments, named expressions, and direct writes to
    ``*.environ["RUNTIME_NAME"]``.
    """
    invalid = [name for name in runtime_globals if not name.isidentifier()]
    if invalid:
        raise ValueError(f"invalid sim runtime global names: {invalid}")

    protected = set(runtime_globals)

    class _ProtectRuntimeGlobals(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if (
                node.id in protected
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                return ast.copy_location(
                    ast.Name(
                        id=f"__evo_ignored_runtime_assignment_{node.id}",
                        ctx=node.ctx,
                    ),
                    node,
                )
            return node

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            node = self.generic_visit(node)
            if (
                isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in protected
            ):
                node.slice = ast.copy_location(
                    ast.Constant(
                        value=(
                            "__EVO_IGNORED_RUNTIME_ASSIGNMENT_"
                            f"{node.slice.value}"
                        )
                    ),
                    node.slice,
                )
            return node

    try:
        parsed = ast.parse(code)
    except SyntaxError:
        # Preserve the worker's syntax-error reporting path. Invalid code
        # cannot execute an overwrite, so the raw source is safe here.
        protected_code = code
    else:
        protected_code = ast.unparse(
            ast.fix_missing_locations(_ProtectRuntimeGlobals().visit(parsed))
        )

    prelude_lines = ["import os as __evo_runtime_os"]
    for name, value in sorted(runtime_globals.items()):
        prelude_lines.append(
            f"__evo_runtime_os.environ[{name!r}] = {value!r}"
        )
        prelude_lines.append(f"{name} = {value!r}")
    prelude = "\n".join(prelude_lines)
    return prelude + "\n" + protected_code


@dataclass
class ResponsesRunOutcome:
    final_output: str
    response_id: str | None
    usage_summary: dict[str, Any]
    artifact_path: Path | None


class EmptyResponsesOutputError(RuntimeError):
    """The provider returned a schema-invalid Responses object.

    OpenAI Agents SDK expects ``response.output`` to be a list. Some long
    Alaya requests have instead returned ``null``; without this guard the SDK
    raises an opaque Pydantic ``ModelResponse.output`` validation error and
    discards the response metadata that is needed to diagnose the backend.
    """


def _bounded_text(
    text: str,
    *,
    max_chars: int = MAX_TEXT_TOOL_OUTPUT_CHARS,
    label: str = "tool output",
) -> str:
    """Keep tool results within the model context while preserving the tail."""
    if len(text) <= max_chars:
        return text
    marker = (
        f"\n... [{label} truncated: {len(text):,} chars total; "
        f"showing head and tail] ...\n"
    )
    remaining = max(max_chars - len(marker), 2)
    head = remaining * 2 // 3
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


class GuardedOpenAIResponsesModel(OpenAIResponsesModel):
    """OpenAI Responses model with explicit diagnostics for ``output=null``."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        *,
        invalid_response_path: Path | None = None,
    ) -> None:
        super().__init__(model, openai_client)
        self.invalid_response_path = invalid_response_path

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        response = await super()._fetch_response(*args, **kwargs)
        if kwargs.get("stream", False) or getattr(response, "output", None) is not None:
            return response

        try:
            raw = response.model_dump(mode="json", exclude_none=False)
        except Exception:
            raw = {"repr": repr(response)}
        diagnostic = {
            "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": str(self.model),
            "response": raw,
        }
        if self.invalid_response_path is not None:
            try:
                self.invalid_response_path.parent.mkdir(parents=True, exist_ok=True)
                with self.invalid_response_path.open("a") as fp:
                    fp.write(json.dumps(diagnostic, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        raise EmptyResponsesOutputError(
            "Responses provider returned output=null "
            f"(model={self.model}, id={getattr(response, 'id', None)!r}, "
            f"status={getattr(response, 'status', None)!r}, "
            f"error={getattr(response, 'error', None)!r}, "
            f"incomplete_details={getattr(response, 'incomplete_details', None)!r}); "
            "this is a training-system/API failure, not a Generator result"
        )


PROVIDER_ENV = "EVO_RESPONSES_PROVIDER"   # openrouter (default) | local_cpa
LOCAL_CPA_URL_ENV = "LOCAL_CPA_URL"
LOCAL_CPA_KEY_ENV = "LOCAL_CPA_KEY"


def _provider() -> str:
    """Explicit provider selector; `local_cpa` routes to LOCAL_CPA_URL/KEY."""
    name = os.environ.get(PROVIDER_ENV, "").strip().lower() or "openrouter"
    if name not in ("openrouter", "local_cpa"):
        raise RuntimeError(f"{PROVIDER_ENV} must be 'openrouter' or 'local_cpa', got {name!r}")
    return name


def _responses_base_url() -> str:
    if _provider() == "local_cpa":
        raw = os.environ.get(LOCAL_CPA_URL_ENV, "").strip()
        if not raw:
            raise RuntimeError(f"{PROVIDER_ENV}=local_cpa requires {LOCAL_CPA_URL_ENV}")
        parts = urlsplit(raw.rstrip("/"))
        path = parts.path.rstrip("/")
        if not path.endswith("/v1"):
            path += "/v1"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    # OPENAI_API_BASE is the key the shared credential file uses for the
    # Responses provider, and it wins: a workstation `.env` typically also
    # carries an OPENAI_BASE_URL for some other (chat-completions) provider,
    # and silently preferring that would pair one provider's root with
    # another's key.
    raw = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or DEFAULT_RESPONSES_BASE_URL
    ).rstrip("/")
    # The existing repo .env uses ANTHROPIC_BASE_URL=https://... with no
    # version suffix. OpenAI clients expect the API root including /v1.
    parts = urlsplit(raw)
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _is_openrouter_url(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _responses_api_key(base_url: str) -> str | None:
    """Select credentials for the configured provider without key aliasing."""
    if _provider() == "local_cpa":
        return os.environ.get(LOCAL_CPA_KEY_ENV) or None
    if _is_openrouter_url(base_url):
        # OPENROUTER_EVO_KEY is the dedicated experiment-spend key; fall back
        # to the general OpenRouter key so other hosts keep working.
        return (
            os.environ.get("OPENROUTER_EVO_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")


def _provider_model_name(model_name: str, base_url: str) -> str:
    """Translate internal OpenAI model aliases to OpenRouter model IDs."""
    if _is_openrouter_url(base_url) and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


TIMEOUT_ENV = "EVO_RESPONSES_TIMEOUT_S"
# Per-request timeout by reasoning effort. Runner.run is non-streaming, so a
# call that the gateway silently drops costs the whole timeout before the
# transient retry kicks in: at 600 s four dropped calls added a 9-minute tail to
# a round in run 11 whose longest real executor call was 218 s (generator at
# high effort: 382 s).
_TIMEOUT_BY_EFFORT_S = {"minimal": 300.0, "low": 300.0, "medium": 300.0}
_DEFAULT_TIMEOUT_S = 600.0


MAX_RETRIES_ENV = "EVO_RESPONSES_MAX_RETRIES"
# Request-level retries inside the OpenAI client (timeouts, connection errors,
# 408/409/429/5xx). Retrying the one call keeps the executor session and its
# tool state; the loop's transient retry, which restarts the whole session,
# remains the fallback. Gateway evidence (cliproxyapi main.log, 2026-09-24/25):
# 24 of ~1200 first-of-session requests got no upstream byte in 10 min and
# only ended when this client hung up (gateway status 499); a normal first
# call takes 3 s (p99 14 s).
_DEFAULT_MAX_RETRIES = 2


def _max_retries() -> int:
    return int(os.environ.get(MAX_RETRIES_ENV, _DEFAULT_MAX_RETRIES))


FIRST_TIMEOUT_ENV = "EVO_RESPONSES_FIRST_TIMEOUT_S"
# The hangs are almost exclusively the FIRST request of a session (the big
# uncached prompt): gateway log 2026-09-24/25, 23 of 1207 first requests vs
# 1 of 6531 follow-ups, while a normal first request takes 3.4 s (p99 13.5 s,
# max 18 s at low effort). Cap that one call tightly at low/medium effort; a
# high-effort generator's first call reasons over the whole context and keeps
# the effort timeout.
_FIRST_TIMEOUT_BY_EFFORT_S = {"minimal": 90.0, "low": 90.0, "medium": 90.0}


def _first_request_timeout(effort: str | None) -> float | None:
    override = os.environ.get(FIRST_TIMEOUT_ENV)
    if override:
        value = float(override)
        return value if value > 0 else None
    return _FIRST_TIMEOUT_BY_EFFORT_S.get((effort or DEFAULT_AGENT_EFFORT).lower())


class SessionClient(AsyncOpenAI):
    """AsyncOpenAI whose first logical request gets its own (shorter) timeout.

    `request()` is one logical call; the client's own retry loop (max_retries)
    runs inside it, so a timed-out first call is re-sent with the same short
    timeout instead of the session dying.
    """

    def __init__(self, *args, first_request_timeout: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_request_timeout = first_request_timeout
        self._logical_requests = 0

    async def request(self, cast_to, options, *, stream=False, stream_cls=None):
        self._logical_requests += 1
        if (
            self._logical_requests == 1
            and self._first_request_timeout
            and getattr(options, "timeout", None) in (None, NOT_GIVEN)
        ):
            options = options.model_copy(update={"timeout": self._first_request_timeout})
        return await super().request(cast_to, options, stream=stream, stream_cls=stream_cls)


def _request_timeout(effort: str | None) -> float:
    override = os.environ.get(TIMEOUT_ENV)
    if override:
        return float(override)
    return _TIMEOUT_BY_EFFORT_S.get((effort or DEFAULT_AGENT_EFFORT).lower(), _DEFAULT_TIMEOUT_S)


def _client(effort: str | None = None) -> AsyncOpenAI:
    # The shared provider credentials live outside the repository (~/.env, or
    # ~/work/.env on DSW), while the repository .env contains dataset/runtime
    # settings. Keep direct library callers correct as well as the normal loop
    # entrypoint.
    from evo.engine.envfile import load_env_chain

    load_env_chain()
    base_url = _responses_base_url()
    api_key = _responses_api_key(base_url)
    if not api_key:
        raise RuntimeError(
            "Responses backend requires LOCAL_CPA_KEY when "
            f"{PROVIDER_ENV}=local_cpa, OPENROUTER_EVO_KEY (or OPENROUTER_API_KEY) "
            "for OpenRouter, or OPENAI_API_KEY/ANTHROPIC_AUTH_TOKEN for another provider"
        )
    return SessionClient(base_url=base_url, api_key=api_key,
                         timeout=_request_timeout(effort), max_retries=_max_retries(),
                         first_request_timeout=_first_request_timeout(effort))


def _settings(effort: str | None) -> ModelSettings:
    # Responses has no Claude /fast toggle. Explicitly select the default
    # service tier and never ask the provider for a priority/fast lane.
    return ModelSettings(
        reasoning={"effort": effort or DEFAULT_AGENT_EFFORT},
        store=False,
        parallel_tool_calls=True,
        extra_args={"service_tier": DEFAULT_SERVICE_TIER},
    )


def _is_within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _resolve_scoped(path: str, *, cwd: Path, roots: list[Path]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve()
    if not _is_within(resolved, roots):
        raise PermissionError(f"path is outside allowed roots: {path}")
    return resolved


# The Responses vision input accepts exactly these; anything else that
# mimetypes calls image/* (svg+xml, tiff, bmp, x-icon) is rejected with a 400.
_VISION_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_MAGIC = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/webp": (b"RIFF",),
}


def _looks_like(mime: str, data: bytes) -> bool:
    """Cheap magic-byte check so a truncated render is never sent as an image."""
    if len(data) < 16:
        return False
    if mime == "image/webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return any(data.startswith(m) for m in _MAGIC[mime])


def make_file_tools(
    *,
    cwd: Path,
    read_roots: list[Path],
    write_root: Path | None,
) -> list[Any]:
    cwd = cwd.resolve()
    roots = list(dict.fromkeys([cwd, *(p.resolve() for p in read_roots)]))
    writable = write_root.resolve() if write_root is not None else None

    @function_tool(name_override="Read")
    async def read_file(file_path: str, offset: int = 1, limit: int = 2000) -> Any:
        """Read a scoped text file or return an image as a vision tool output."""
        path = _resolve_scoped(file_path, cwd=cwd, roots=roots)
        if not path.is_file():
            raise FileNotFoundError(path)
        mime, _ = mimetypes.guess_type(path.name)
        if mime in _VISION_MIMES:
            data = path.read_bytes()
            if not _looks_like(mime, data):
                # A truncated render or a mislabelled file. Attaching it
                # produces a provider-side 400 ("does not represent a valid
                # image") that abort a training round (run 8, round 5).
                return [ToolOutputText(text=(
                    f"{path.name} is not a valid {mime} image "
                    f"({len(data):,} bytes, header {data[:8]!r}); it was not attached. "
                    "Re-render it before reading."
                ))]
            encoded = base64.b64encode(data).decode("ascii")
            return [ToolOutputImage(image_url=f"data:{mime};base64,{encoded}", detail="high")]
        # Other image/* types (SVG, TIFF, BMP …) are not accepted by the vision
        # input; SVG is XML and reads fine as text below.
        if path.name.endswith(".responses.json"):
            return [ToolOutputText(text=(
                f"Raw Responses trajectory blocked ({path.stat().st_size:,} bytes). "
                "Read the compact executor.log instead, or use Grep with a "
                "targeted error/tool pattern."
            ))]
        lines = path.read_text(errors="replace").splitlines()
        start = max(offset - 1, 0)
        selected = lines[start : start + max(limit, 1)]
        numbered = "\n".join(f"{start+i+1:6d}\t{line}" for i, line in enumerate(selected))
        return [ToolOutputText(text=_bounded_text(numbered, label=f"Read {path.name}"))]

    @function_tool(name_override="Glob")
    async def glob_files(pattern: str, path: str = ".") -> str:
        """List files matching a glob below an allowed root."""
        base = _resolve_scoped(path, cwd=cwd, roots=roots)
        if not base.is_dir():
            raise NotADirectoryError(base)
        matches = [p.resolve() for p in base.glob(pattern)]
        safe = sorted(str(p) for p in matches if _is_within(p, roots))
        return "\n".join(safe[:2000]) or "(no matches)"

    @function_tool(name_override="Grep")
    async def grep_files(pattern: str, path: str = ".", file_glob: str = "*") -> str:
        """Regex-search scoped text files and return path:line:content matches."""
        base = _resolve_scoped(path, cwd=cwd, roots=roots)
        rx = re.compile(pattern)
        files = [base] if base.is_file() else base.rglob(file_glob)
        found: list[str] = []
        for fp in files:
            if len(found) >= MAX_GREP_MATCHES or not fp.is_file():
                break
            resolved = fp.resolve()
            if not _is_within(resolved, roots):
                continue
            try:
                for lineno, line in enumerate(fp.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        safe_line = _bounded_text(
                            line,
                            max_chars=MAX_GREP_LINE_CHARS,
                            label="matching line",
                        )
                        found.append(f"{resolved}:{lineno}:{safe_line}")
                        if len(found) >= MAX_GREP_MATCHES:
                            break
            except OSError:
                continue
        return _bounded_text("\n".join(found), label="Grep results") or "(no matches)"

    tools: list[Any] = [read_file, glob_files, grep_files]
    if writable is None:
        return tools

    @function_tool(name_override="Write")
    async def write_file(file_path: str, content: str) -> str:
        """Write a UTF-8 file under the run's writable root."""
        path = _resolve_scoped(file_path, cwd=cwd, roots=[writable])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    @function_tool(name_override="Edit")
    async def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace an exact string in a UTF-8 file under the writable root."""
        path = _resolve_scoped(file_path, cwd=cwd, roots=[writable])
        text = path.read_text()
        count = text.count(old_string)
        if count == 0:
            raise ValueError("old_string was not found")
        if count > 1 and not replace_all:
            raise ValueError(f"old_string occurs {count} times; set replace_all=true")
        updated = text.replace(old_string, new_string, -1 if replace_all else 1)
        path.write_text(updated)
        return f"edited {path} ({count if replace_all else 1} replacement(s))"

    tools.extend([write_file, edit_file])
    return tools


class SimResponsesMCPServer(MCPServer):
    """In-process MCP adapter around the persistent sim-worker socket."""

    def __init__(
        self,
        port: int,
        runtime_globals: dict[str, str] | None = None,
    ):
        super().__init__(require_approval="never", failure_error_function=None)
        self.port = port
        self.runtime_globals = dict(runtime_globals or {})

    @property
    def name(self) -> str:
        return "sim"

    async def connect(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_tools(self, run_context=None, agent=None) -> list[Tool]:
        return [Tool(
            name="mcp__sim__execute_python_code",
            description=(
                "Execute Python code in a persistent Python session with the "
                "frozen `assembly` SDK preloaded and the immutable runtime "
                "globals (ASSEMBLY_SDK_DIR, SKILL_DIR, INPUT_DIR, OUTPUT_DIR) "
                "injected. Variables, imports, and helper functions persist "
                "across calls. Returns captured stdout/stderr plus the value "
                "of a trailing expression or a variable named `result`. "
                "Plain interpreter, not a Jupyter kernel: `display()`, IPython "
                "helpers, and `%` magics do not exist; use print() and write "
                "images to files (view PNGs with the Read tool). Each session has "
                "a resident-memory cap (default 6 GB); a snippet that crosses it "
                "is interrupted with a MemoryError. "
            ),
            inputSchema={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        )]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        if tool_name != "mcp__sim__execute_python_code":
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"unknown tool: {tool_name}")],
            )
        code = str((arguments or {}).get("code", ""))
        if self.runtime_globals:
            # Defense in depth; the sim worker also re-injects these values
            # before every execution.
            code = _inject_runtime_globals(code, self.runtime_globals)
        try:
            resp = await asyncio.to_thread(
                send_to_sim_worker,
                code,
                self.port,
            )
        except Exception as exc:
            from evo.engine.sim_manager import memory_kill_notice
            notice = memory_kill_notice(self.port)
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=notice or f"socket error: {exc}")],
            )
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
        content = _bounded_text("\n".join(parts) or "(no output)", label="sim output")
        return CallToolResult(
            isError=is_error,
            content=[TextContent(type="text", text=content)],
        )

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        raise KeyError(name)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


async def run_responses_agent(
    *,
    name: str,
    instructions: str,
    prompt: str,
    model: str,
    effort: str | None,
    max_turns: int,
    cwd: Path,
    read_roots: list[Path],
    write_root: Path | None,
    subagent_dir: Path | None = None,
    subagent_model: str | None = None,
    subagent_effort: str | None = None,
    subagent_default_effort: str | None = None,
    sim_port: int | None = None,
    runtime_globals: dict[str, str] | None = None,
    artifact_path: Path | None = None,
    extra_tools: list[Any] | None = None,
) -> ResponsesRunOutcome:
    from evo.engine.agents import _parse_agent_md

    client = _client(effort)
    provider_base_url = _responses_base_url()
    model_cache: dict[str, OpenAIResponsesModel] = {}

    def resolve_model(model_name: str) -> OpenAIResponsesModel:
        provider_model = _provider_model_name(model_name, provider_base_url)
        if provider_model not in model_cache:
            invalid_path = (
                artifact_path.with_name(
                    f"{artifact_path.stem}.invalid_response.jsonl"
                )
                if artifact_path is not None
                else None
            )
            model_cache[provider_model] = GuardedOpenAIResponsesModel(
                provider_model,
                client,
                invalid_response_path=invalid_path,
            )
        return model_cache[provider_model]

    file_tools = make_file_tools(cwd=cwd, read_roots=read_roots, write_root=write_root)
    specialist_tools: list[Any] = []
    specialist_names: list[str] = []
    if subagent_dir is not None and subagent_dir.is_dir():
        for md in sorted(subagent_dir.glob("*.md")):
            sub_name, description, _tools, sub_model, sub_effort, body = _parse_agent_md(
                md.read_text(), md.stem
            )
            specialist = OpenAIAgent(
                name=sub_name,
                instructions=body.strip(),
                model=resolve_model(subagent_model or sub_model or model),
                model_settings=_settings(
                    subagent_effort
                    or sub_effort
                    or subagent_default_effort
                    or effort
                ),
                tools=make_file_tools(cwd=cwd, read_roots=read_roots, write_root=None),
            )
            tool_name = sub_name.replace("-", "_")
            specialist_names.append(tool_name)
            specialist_tools.append(specialist.as_tool(
                tool_name=tool_name,
                tool_description=description or f"Run the {sub_name} specialist",
                max_turns=min(max_turns, VALIDATOR_MAX_TURNS),
            ))

    backend_note = (
        "\n\n# Responses backend tool mapping\n"
        # make_file_tools only registers Write/Edit when there is a writable
        # root, so name the tools that actually exist for this agent.
        + (
            "Use Read, Glob, Grep, Write, and Edit directly. "
            if write_root is not None else "Use Read, Glob, and Grep directly. "
        )
        + "Bash is unavailable. "
        # Only advertise the sim tool when one is actually attached — a judge
        # or other sim-less agent told about it will try to call it and stall.
        + (
            "The sim MCP tool is named mcp__sim__execute_python_code: a "
            "persistent Python session with the frozen `assembly` SDK "
            "preloaded. "
            if sim_port is not None else ""
        )
        + (
            "Claude-style Agent calls are replaced by these specialist tools: "
            + ", ".join(specialist_names)
            + ". Pass the complete specialist task in their input argument."
            if specialist_names else "No specialist tools are registered."
        )
    )
    mcp_servers = (
        [SimResponsesMCPServer(sim_port, runtime_globals)]
        if sim_port is not None
        else []
    )
    agent = OpenAIAgent(
        name=name,
        instructions=instructions + backend_note,
        model=resolve_model(model),
        model_settings=_settings(effort),
        tools=[*file_tools, *specialist_tools, *(extra_tools or [])],
        mcp_servers=mcp_servers,
    )

    try:
        result = await Runner.run(
            agent,
            prompt,
            max_turns=max_turns,
            run_config=RunConfig(
                tracing_disabled=True,
                workflow_name=f"evo-assemble:{name}",
            ),
            # Deliberately omit previous_response_id/conversation_id. The
            # runner carries typed items explicitly, which Alaya supports.
        )
        final = result.final_output
        if not isinstance(final, str):
            final = json.dumps(_jsonable(final), ensure_ascii=False)
        usage = result.context_wrapper.usage
        # Price the run here: the Responses API reports tokens but no money,
        # so without this every downstream cost number reads as $0.00.
        usage_summary = pricing.summarize(usage, model)
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps({
                "backend": "openai-agents-responses",
                "model": _provider_model_name(model, provider_base_url),
                "requested_model": model,
                "provider_base_url": provider_base_url,
                "effort": effort,
                "subagent_model_override": subagent_model,
                "subagent_effort_override": subagent_effort,
                "service_tier": DEFAULT_SERVICE_TIER,
                "store": False,
                "state_strategy": "explicit_items",
                "previous_response_id": None,
                "sim_runtime_globals": sorted((runtime_globals or {}).keys()),
                "response_id": result.last_response_id,
                "final_output": final,
                "usage": _jsonable(usage),
                "usage_summary": usage_summary,
                "pricing_source": pricing.PRICING_SOURCE,
                "items": _jsonable(result.to_input_list()),
            }, indent=2, ensure_ascii=False))
        return ResponsesRunOutcome(
            final_output=final,
            response_id=result.last_response_id,
            usage_summary=usage_summary,
            artifact_path=artifact_path,
        )
    finally:
        await client.close()
