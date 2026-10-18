from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from harness2.cli import judge_call
from harness2.judging import JudgeResult


class JudgeCliTests(unittest.TestCase):
    def test_cli_reads_one_json_request_and_writes_one_result(self):
        result = JudgeResult("PASS", None, "codex_cli", "gpt-6-luna", 1.0)
        with (
            patch("harness2.cli.judge_call.create_judge") as factory,
            patch("sys.stdin", io.StringIO('{"prompt":"check"}')),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch("harness2.cli.judge_call.config.JUDGE_BACKEND", "codex_cli"),
            patch("harness2.cli.judge_call.config.JUDGE_MODEL", "gpt-6-luna"),
        ):
            factory.return_value.judge.return_value = result
            code = judge_call.main([])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["text"], "PASS")
        self.assertEqual(factory.call_args.args[0].backend, "codex_cli")

    def test_check_validates_without_reading_stdin(self):
        with (
            patch("harness2.cli.judge_call.create_judge") as factory,
            patch("harness2.cli.judge_call.config.JUDGE_BACKEND", "api"),
            patch("harness2.cli.judge_call.config.JUDGE_MODEL", "m"),
        ):
            self.assertEqual(judge_call.main(["--check"]), 0)
            factory.return_value.check.assert_called_once_with()
            factory.return_value.judge.assert_not_called()

    def test_cli_argument_overrides_shared_backend(self):
        with (
            patch("harness2.cli.judge_call.create_judge") as factory,
            patch("sys.stdin", io.StringIO('{"prompt":"p"}')),
            patch("sys.stdout", new_callable=io.StringIO),
            patch("harness2.cli.judge_call.config.JUDGE_BACKEND", "api"),
            patch("harness2.cli.judge_call.config.JUDGE_MODEL", "m"),
        ):
            factory.return_value.judge.return_value = JudgeResult("ok", None, "codex_cli", "m", 0.1)
            self.assertEqual(judge_call.main(["--backend", "codex_cli"]), 0)
            self.assertEqual(factory.call_args.args[0].backend, "codex_cli")


if __name__ == "__main__":
    unittest.main()
