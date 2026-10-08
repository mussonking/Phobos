import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_quant as q


class ClosedConsole(io.StringIO):
    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


class QuantSupervisorTests(unittest.TestCase):
    def run_case(self, root, console, rc=0, validation_error=None):
        output = root / "model.gguf"
        recipe = {"output": str(output), "expected_payload_bytes": 10,
                  "argv": ["fake-quantizer"], "type_counts": {"IQ4_XS": 86}}
        proc = Mock(pid=53)
        def finish():
            output.write_bytes(b"fixture payload")
            return rc
        proc.wait.side_effect = finish
        def validate(args):
            print("PASS output validation")
            if validation_error:
                raise validation_error
        with patch.object(q.sys, "argv", ["run_quant", "--recipe", str(root / "recipe.json"),
                                          "--dry-log", str(root / "dry.log")]), \
             patch.object(q, "load_recipe", return_value=recipe), \
             patch.object(q, "verify_dry", side_effect=lambda args: print("PASS dry validation")), \
             patch.object(q, "verify_output", side_effect=validate) as verifier, \
             patch.object(q.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)), \
             patch.object(q.subprocess, "Popen", return_value=proc) as launch, \
             contextlib.redirect_stdout(console):
            try:
                q.main()
            finally:
                replacement = q.sys.stdout
                if replacement is not console:
                    replacement.close()
        launch.assert_called_once()
        self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        verifier.assert_called_once()
        return json.loads((root / "quant-state.json").read_text())

    def test_closed_console_does_not_invalidate_verification(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            state = self.run_case(root, ClosedConsole())
            self.assertEqual(state["status"], "verified")
            self.assertEqual(state["exit_code"], 0)
            self.assertEqual(state["file_bytes"], len(b"fixture payload"))
            log = (root / "quantize.log").read_text()
            self.assertIn("PASS dry validation", log)
            self.assertIn("PASS output validation", log)
            self.assertIn("QUANT_DONE verified", log)

    def test_open_console_reports_verified_completion(self):
        with tempfile.TemporaryDirectory() as d:
            console = io.StringIO()
            state = self.run_case(pathlib.Path(d), console)
            self.assertEqual(state["status"], "verified")
            self.assertIn("QUANT_DONE verified", console.getvalue())

    def test_real_validation_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                self.run_case(root, ClosedConsole(), validation_error=ValueError("truncated tensor"))
            state = json.loads((root / "quant-state.json").read_text())
            self.assertEqual(state["status"], "verification_failed")
            self.assertEqual(state["error"], "truncated tensor")
            self.assertEqual(state["exit_code"], 0)

    def test_quantizer_failure_keeps_actual_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            with self.assertRaisesRegex(SystemExit, "Quantizer failed.*7"):
                self.run_case(root, io.StringIO(), rc=7)
            state = json.loads((root / "quant-state.json").read_text())
            self.assertEqual(state["status"], "quantizer_failed")
            self.assertEqual(state["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
