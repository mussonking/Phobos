import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import watch_quant as w


class DetachedVerifierTests(unittest.TestCase):
    def check_final(self, fail=False):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            output = root / "model.gguf"
            output.write_bytes(b"fixture payload")
            state = {"status": "verification_failed", "error": "[Errno 32] Broken pipe",
                     "pid": 123, "supervisor_pid": 456, "started": 1.0}
            original = json.dumps(state)
            (root / "quant-state.json").write_text(original)
            recipe = {"output": str(output), "type_counts": {"IQ4_XS": 86, "IQ4_NL": 43}}
            with patch.object(w.sys, "argv", ["watch_quant", "--recipe", str(root/"recipe.json")]), \
                 patch.object(w, "load_recipe", return_value=recipe), \
                 patch.object(w, "alive", return_value=False), \
                 patch.object(w, "verify_output", side_effect=ValueError("truncated") if fail else None) as verify, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.main(), 1 if fail else 0)
            verify.assert_called_once()
            final = json.loads((root / "quant-final.json").read_text())
            self.assertEqual(final["status"], "verification_failed" if fail else "verified")
            self.assertIsNone(final["exit_code"])
            self.assertEqual((root / "quant-state.json").read_text(), original)
            if fail:
                self.assertEqual(final["error"], "truncated")
            else:
                self.assertEqual(final["file_bytes"], len(b"fixture payload"))

    def test_supervisor_broken_pipe_does_not_invalidate_artifact(self):
        self.check_final()

    def test_real_verification_failure_stays_failure(self):
        self.check_final(fail=True)

    def test_never_replaces_prior_final_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "quant-state.json").write_text(json.dumps({"pid": 123}))
            (root / "quant-final.json").write_text("existing result")
            with patch.object(w.sys, "argv", ["watch_quant", "--recipe", str(root/"recipe.json")]), \
                 patch.object(w, "load_recipe", return_value={}):
                with self.assertRaises(FileExistsError):
                    w.main()
            self.assertEqual((root / "quant-final.json").read_text(), "existing result")


if __name__ == "__main__":
    unittest.main()
