import argparse
import contextlib
import hashlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import quant_recipe as q
from gguf_reader import TensorInfo


class QuantRecipeTests(unittest.TestCase):
    def test_parse_changed_and_retained_types(self):
        text = ("[   1/1412] blk.3.ffn_gate_exps.weight - [4096,2048,288], type = bf16, "
                "size = 4608.00 MiB -> 1224.00 MiB (iq4_xs)\n"
                "[   2/1412] blk.0.hc_attn_fn.weight - [16384,24], type = bf16, "
                "size = 0.750 MiB\n")
        self.assertEqual(q.parse_dry_types(text), {
            "blk.3.ffn_gate_exps.weight": "IQ4_XS", "blk.0.hc_attn_fn.weight": "BF16"})

    def test_duplicate_dry_tensor_rejected(self):
        line = "[1/2] output.weight - [4096,154880], type = bf16, size = 1 MiB\n"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            q.parse_dry_types(line + line)

    def recipe(self, root):
        types = root / "tensor-types.txt"
        types.write_text("^output\\.weight$=q8_0\n")
        recipe = {"format": "glm-iq4-recipe-v1",
                  "argv": ["llama-quantize", "--tensor-type-file", str(types)],
                  "overrides_sha256": hashlib.sha256(types.read_bytes()).hexdigest(),
                  "tensors": [{"name": "output.weight", "target_type": "Q8_0"}]}
        path = root / "recipe.json"
        path.write_text(json.dumps(recipe))
        return path, types

    def test_verification_complete_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            recipe, _ = self.recipe(root)
            log = root / "dry.log"
            log.write_text("[1/1] output.weight - [4096,154880], type = bf16, "
                           "size = 1 MiB -> 0.5 MiB (q8_0)\nquant size = 0.5 MiB\n")
            args = argparse.Namespace(recipe=recipe, log=log)
            with contextlib.redirect_stdout(io.StringIO()):
                q.verify_dry(args)
            log.write_text(log.read_text().replace("quant size = 0.5 MiB\n", ""))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                q.verify_dry(args)
            log.write_text("quant size = 0.5 MiB\n")
            with self.assertRaisesRegex(ValueError, "MISSING"):
                q.verify_dry(args)

    def test_override_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            recipe, types = self.recipe(pathlib.Path(d))
            types.write_text("^output\\.weight$=iq3_s\n")
            with self.assertRaisesRegex(ValueError, "changed after planning"):
                q.load_recipe(recipe)

    def test_oracle_nonexpert_lower_quant_rejected(self):
        with self.assertRaisesRegex(ValueError, "not F32/Q8_0"):
            q.wanted_type(TensorInfo("blk.0.attn_q.weight", [4096,8192], 12, "Q4_K", 0))
        self.assertEqual(q.wanted_type(TensorInfo("blk.45.ffn_down_exps.weight",
                                                [2048,4096,288], 14, "Q6_K", 0)), "IQ4_NL")

    def test_mainline_acceptance_provenance_and_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            oracle = {}
            for il in range(3, 46):
                for part in ("gate", "up", "down"):
                    name = f"blk.{il}.ffn_{part}_exps.weight"
                    dims = [2048,4096,288] if part == "down" else [4096,2048,288]
                    oracle[name] = TensorInfo(name, dims, 12, "Q4_K", 0)
            name = "blk.0.hc_attn_fn.weight"
            oracle[name] = TensorInfo(name, [16384,24], 8, "Q8_0", 0)
            source = {n: TensorInfo(n,t.shape,30,"BF16",0) for n,t in oracle.items()}
            dry = root / "dry.log"
            def write_dry(nonexpert_type="bf16"):
                lines = []
                for i, t in enumerate(oracle.values()):
                    target = q.wanted_type(t).lower() if t.name != name else nonexpert_type
                    lines.append(f"[{i+1}/{len(oracle)}] {t.name} - [4096,2048], type = bf16, "
                                 f"size = 1 MiB -> 0.5 MiB ({target})")
                dry.write_text("\n".join(lines) + "\nquant size = 1 MiB\n")
            args = argparse.Namespace(oracle=root/"oracle.gguf", source=root/"source.gguf",
                imatrix=root/"imatrix", quantizer=root/"llama-quantize", output=root/"out.gguf",
                out_dir=root/"plan", threads=12, accept_mainline_bf16=dry)
            def read_model(path):
                return (oracle if path == args.oracle else source), []
            write_dry()
            with patch.object(q, "read_model", side_effect=read_model), contextlib.redirect_stdout(io.StringIO()):
                q.plan(args)
                recipe = json.loads((args.out_dir/"recipe.json").read_text())
                self.assertEqual(recipe["mainline_exclusions"]["tensors"], [name])
                self.assertEqual(recipe["mainline_exclusions"]["extra_trunk_bytes_per_token"], 368640)
                write_dry("f16")
                with self.assertRaisesRegex(ValueError, "cannot accept"):
                    q.plan(args)
                write_dry()
                oracle[name].shape = [4096,160000]
                source[name].shape = [4096,160000]
                with self.assertRaisesRegex(ValueError, "exceeds 0.5 GB"):
                    q.plan(args)


if __name__ == "__main__":
    unittest.main()
