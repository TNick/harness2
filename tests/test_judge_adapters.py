from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness2.judging import JudgeError, JudgeRequest, JudgeSettings, create_judge
from harness2.judging.adapters import ApiJudge, CodexCliJudge


class JudgeAdapterTests(unittest.TestCase):
    def test_factory_retains_api_default_and_adds_codex(self):
        self.assertIsInstance(create_judge(JudgeSettings("api", "m")), ApiJudge)
        self.assertIsInstance(create_judge(JudgeSettings("codex_cli", "m")), CodexCliJudge)
        with self.assertRaises(JudgeError):
            create_judge(JudgeSettings("claude", "m"))

    @patch("harness2.judging.adapters.urlopen")
    def test_api_sends_prompt_effort_schema_and_returns_text(self, urlopen):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": '{"passed":true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        ).encode()
        urlopen.return_value = response
        schema = {
            "type": "object",
            "properties": {"passed": {"type": "boolean"}},
            "required": ["passed"],
        }
        judge = ApiJudge(
            JudgeSettings(
                "api",
                "gpt-6-luna",
                effort="medium",
                api_base_url="https://example.test/v1",
                api_key="secret",
            )
        )
        with patch.dict("sys.modules", {"jsonschema": __import__("jsonschema")}):
            got = judge.judge(JudgeRequest("evaluate", "system", schema))
        req = urlopen.call_args.args[0]
        sent = json.loads(req.data)
        self.assertEqual(req.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(sent["reasoning_effort"], "medium")
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(sent["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(got.parsed, {"passed": True})
        self.assertEqual(got.usage["prompt_tokens"], 12)

    def test_api_validation_never_falls_back_to_codex(self):
        judge = ApiJudge(JudgeSettings("api", "m"))
        with self.assertRaisesRegex(JudgeError, "BASE_URL"):
            judge.check()

    @patch("harness2.judging.adapters.shutil.which", return_value="/usr/bin/codex")
    @patch("harness2.judging.adapters.subprocess.run")
    def test_codex_check_uses_local_login_status(self, run, _which):
        run.side_effect = [
            MagicMock(returncode=0, stdout="--json --output-last-message --disable", stderr=""),
            MagicMock(returncode=0, stdout="Logged in", stderr=""),
        ]
        create_judge(JudgeSettings("codex_cli", "gpt-6-luna")).check()
        self.assertEqual(run.call_args_list[0].args[0], ["/usr/bin/codex", "exec", "--help"])
        self.assertEqual(run.call_args_list[1].args[0], ["/usr/bin/codex", "login", "status"])

    @patch("harness2.judging.adapters.shutil.which", return_value="/usr/bin/codex")
    @patch("harness2.judging.adapters.subprocess.run")
    @patch("harness2.judging.adapters.subprocess.Popen")
    def test_codex_uses_ephemeral_cwd_stdin_and_authoritative_output(self, popen, run, _which):
        run.side_effect = [
            MagicMock(returncode=0, stdout="--json --output-last-message --disable", stderr=""),
            MagicMock(returncode=0, stdout="Logged in", stderr=""),
        ]
        proc = MagicMock(returncode=0)

        def communicate(prompt, timeout):
            args = popen.call_args.args[0]
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text("PASS", encoding="utf-8")
            proc.returncode = 0
            return ('{"type":"turn.completed"}\n', "")

        proc.communicate.side_effect = communicate
        popen.return_value = proc
        judge = CodexCliJudge(JudgeSettings("codex_cli", "gpt-6-luna", effort="medium"))
        got = judge.judge(JudgeRequest("prompt", "rules"))
        args, kwargs = popen.call_args
        self.assertIn("--ephemeral", args[0])
        self.assertIn("--ignore-user-config", args[0])
        self.assertIn("--disable", args[0])
        self.assertIn("shell_tool", args[0])
        self.assertNotIn("code_mode_host", args[0])
        self.assertEqual(args[0][-1], "-")
        self.assertEqual(kwargs["stdin"], __import__("subprocess").PIPE)
        self.assertEqual(got.text, "PASS")
        self.assertEqual(got.backend, "codex_cli")

    @patch("harness2.judging.adapters.shutil.which", return_value="/usr/bin/codex")
    @patch("harness2.judging.adapters.subprocess.run")
    @patch("harness2.judging.adapters.subprocess.Popen")
    def test_codex_tool_events_are_rejected_with_safe_type_diagnostic(self, popen, run, _which):
        run.side_effect = [
            MagicMock(returncode=0, stdout="--json --output-last-message --disable", stderr=""),
            MagicMock(returncode=0, stdout="Logged in", stderr=""),
        ]
        proc = MagicMock(returncode=0)
        proc.communicate.return_value = (
            '{"type":"item.completed","item":{"type":"command_execution"}}\n'
            '{"type":"turn.completed"}\n',
            "",
        )
        popen.return_value = proc
        with self.assertRaisesRegex(JudgeError, "command_execution"):
            CodexCliJudge(JudgeSettings("codex_cli", "gpt-6-luna")).judge(JudgeRequest("prompt"))

    def test_empty_and_malformed_structured_output_fail(self):
        from harness2.judging.adapters import _result

        s = JudgeSettings("codex_cli", "m")
        with self.assertRaises(JudgeError):
            _result(" ", s, 0, backend="codex_cli")
        with patch.dict("sys.modules", {"jsonschema": __import__("jsonschema")}):
            with self.assertRaises(JudgeError):
                _result("not json", s, 0, backend="codex_cli", schema={"type": "object"})


if __name__ == "__main__":
    unittest.main()
