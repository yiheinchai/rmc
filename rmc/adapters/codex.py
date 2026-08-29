"""Codex backend: ``codex exec``, using its native ``--output-schema``."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from . import AgentResult, Session, extract_json
from ._proc import run_cmd, which


class CodexAdapter:
    name = "codex"

    def __init__(self, *, model: str | None = None, binary: str = "codex") -> None:
        self.model = model
        self.binary = binary

    def available(self) -> bool:
        return which(self.binary) is not None

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        cwd: Path | None = None,
        schema: dict[str, Any] | None = None,
        tools: bool = False,
        timeout: int = 180,
        session: Session | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        if not self.available():
            return AgentResult(ok=False, error=f"{self.binary} not on PATH", backend=self.name)

        tmpdir = Path(tempfile.mkdtemp(prefix="rmc-codex-"))
        last_msg = tmpdir / "last.txt"

        schema_file: Path | None = None
        if schema:
            schema_file = tmpdir / "schema.json"
            schema_file.write_text(json.dumps(_strict(schema)), encoding="utf-8")

        argv = build_argv(
            binary=self.binary,
            model=self.model,
            session=session,
            cwd=cwd,
            schema_path=schema_file,
            output_path=last_msg,
            tools=tools,
            allowed_tools=allowed_tools,
        )

        full_prompt = prompt if not system else f"{system}\n\n---\n\n{prompt}"

        code, out, err, dur = run_cmd(argv, cwd=cwd, timeout=timeout, stdin=full_prompt)

        text = ""
        if last_msg.exists():
            text = last_msg.read_text(encoding="utf-8").strip()
        if not text:
            text = _last_agent_message(out) or out.strip()

        if code != 0 and not text:
            return AgentResult(
                ok=False,
                error=(err or f"exit {code}").strip()[:2000],
                duration_s=dur,
                backend=self.name,
                raw=out[:4000],
            )

        data = extract_json(text) if schema else None
        if schema and data is None:
            return AgentResult(
                ok=False,
                text=text,
                error="model did not return parseable JSON",
                duration_s=dur,
                backend=self.name,
                raw=out[:4000],
            )
        return AgentResult(
            ok=True,
            text=text,
            data=data,
            duration_s=dur,
            backend=self.name,
            raw=out[:4000],
        )


def build_argv(
    *,
    binary: str = "codex",
    model: str | None = None,
    session: Session | None = None,
    cwd: Path | None = None,
    schema_path: Path | None = None,
    output_path: Path | None = None,
    tools: bool = False,
    allowed_tools: list[str] | None = None,
    prompt: str | None = None,
) -> list[str]:
    """Assemble a ``codex exec`` argv for one-shot, resume, or fork invocations."""
    sandbox = _sandbox_mode(tools, allowed_tools)
    opts: list[str] = ["--skip-git-repo-check", "--ephemeral"]
    if model:
        opts += ["-m", model]
    if output_path is not None:
        opts += ["-o", str(output_path)]
    if schema_path is not None:
        opts += ["--output-schema", str(schema_path)]

    if session and session.resume:
        # Branch a throwaway child from the live session, mirroring
        # `claude --resume ID --fork-session`.
        argv = [binary, "exec", "fork", session.id, *opts]
        if sandbox != "read-only":
            argv += ["-c", f'sandbox_mode="{sandbox}"']
        if prompt is not None:
            argv.append(prompt)
        return argv

    if session:
        argv = [binary, "exec", "resume", session.id, *opts]
        if sandbox != "read-only":
            argv += ["-c", f'sandbox_mode="{sandbox}"']
        if prompt is not None:
            argv.append(prompt)
        return argv

    argv = [binary, "exec", *opts, "-s", sandbox]
    if cwd:
        argv += ["-C", str(cwd)]
    return argv


def reflection_fork_argv(
    session_id: str,
    prompt: str,
    *,
    binary: str = "codex",
    model: str | None = None,
) -> list[str]:
    """Argv for a background reflection fork that can run ``rmc add``."""
    argv = [
        binary,
        "exec",
        "fork",
        session_id,
        "--ephemeral",
        "--skip-git-repo-check",
        "-c",
        'sandbox_mode="workspace-write"',
    ]
    if model:
        argv += ["-m", model]
    argv.append(prompt)
    return argv


def _sandbox_mode(tools: bool, allowed_tools: list[str] | None) -> str:
    # Selection passes `allowed_tools` without `tools=True`; both mean the
    # agent needs to search the store with shell commands.
    if tools or allowed_tools:
        return "workspace-write"
    return "read-only"


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Codex's schema mode is happier with explicit additionalProperties."""
    out = dict(schema)
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
        props = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: _strict(v) if isinstance(v, dict) else v for k, v in props.items()}
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = _strict(out["items"])
    return out


def _last_agent_message(stdout: str) -> str:
    """Recover the final message from ``--json`` JSONL, if -o produced nothing."""
    best = ""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        msg = row.get("msg") if isinstance(row.get("msg"), dict) else row
        if not isinstance(msg, dict):
            continue
        if msg.get("type") in ("agent_message", "assistant_message", "item.completed"):
            for key in ("message", "text", "content"):
                val = msg.get(key)
                if isinstance(val, str) and val.strip():
                    best = val.strip()
    return best
