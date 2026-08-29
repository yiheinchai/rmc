"""Unit tests for Codex adapter session fork and sandbox argv."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from rmc.adapters import Session
from rmc.adapters.codex import CodexAdapter, build_argv, reflection_fork_argv
from rmc import select_agent


SESSION = "0e7c1a42-1f3b-4c0d-9a55-2b8e6d4f10aa"


class TestCodexArgv(unittest.TestCase):
    def test_ephemeral_exec_is_read_only_by_default(self) -> None:
        argv = build_argv()
        self.assertEqual(argv[:4], ["codex", "exec", "--skip-git-repo-check", "--ephemeral"])
        self.assertIn("-s", argv)
        self.assertEqual(argv[argv.index("-s") + 1], "read-only")

    def test_tools_flag_enables_workspace_write(self) -> None:
        argv = build_argv(tools=True)
        self.assertEqual(argv[argv.index("-s") + 1], "workspace-write")

    def test_allowed_tools_enables_workspace_write(self) -> None:
        argv = build_argv(allowed_tools=["Grep", "Bash(grep:*)"])
        self.assertEqual(argv[argv.index("-s") + 1], "workspace-write")

    def test_fork_uses_exec_fork_subcommand(self) -> None:
        argv = build_argv(session=Session(id=SESSION, resume=True))
        self.assertEqual(argv[:5], ["codex", "exec", "fork", SESSION, "--skip-git-repo-check"])

    def test_fork_with_search_gets_workspace_write_via_config(self) -> None:
        argv = build_argv(
            session=Session(id=SESSION, resume=True),
            allowed_tools=["Grep"],
        )
        self.assertIn("-c", argv)
        idx = argv.index("-c")
        self.assertEqual(argv[idx + 1], 'sandbox_mode="workspace-write"')
        self.assertNotIn("-s", argv)

    def test_resume_uses_exec_resume_subcommand(self) -> None:
        argv = build_argv(session=Session(id=SESSION, resume=False))
        self.assertEqual(argv[:5], ["codex", "exec", "resume", SESSION, "--skip-git-repo-check"])

    def test_reflection_fork_argv_uses_workspace_write(self) -> None:
        argv = reflection_fork_argv(SESSION, "reflect now")
        self.assertEqual(argv[:4], ["codex", "exec", "fork", SESSION])
        self.assertIn('sandbox_mode="workspace-write"', argv)
        self.assertEqual(argv[-1], "reflect now")


class TestCodexAdapterRun(unittest.TestCase):
    def test_run_passes_fork_argv_when_session_resumes(self) -> None:
        adapter = CodexAdapter()
        seen: list[list[str]] = []

        def fake_run(argv, **kwargs):
            seen.append(argv)
            return 0, "", "", 0.1

        with patch.object(adapter, "available", return_value=True), patch(
            "rmc.adapters.codex.run_cmd", side_effect=fake_run
        ):
            adapter.run(
                "pick lessons",
                session=Session(id=SESSION, resume=True),
                allowed_tools=["Grep"],
            )

        self.assertEqual(len(seen), 1)
        self.assertIn("fork", seen[0])
        self.assertIn(SESSION, seen[0])


class TestSelectAgentCodex(unittest.TestCase):
    def test_codex_can_fork_when_available(self) -> None:
        from rmc.store import Store
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = Store.init(Path(tmp))
            adapter = CodexAdapter()
            with patch.object(adapter, "available", return_value=True):
                ok, why = select_agent.available(store, adapter, SESSION)
            self.assertTrue(ok, why)

    def test_mock_still_cannot_fork(self) -> None:
        from rmc.adapters.mock import MockAdapter
        from rmc.store import Store
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = Store.init(Path(tmp))
            ok, why = select_agent.available(store, MockAdapter(), SESSION)
            self.assertFalse(ok)
            self.assertIn("fork", why)


if __name__ == "__main__":
    unittest.main()
