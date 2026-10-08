"""Small count-only GGUF fixtures and sparse importance metadata, standard library only."""
import hashlib
import json
import math
import pathlib
import struct
import tempfile
import unittest
from unittest.mock import patch

import make_profile as P
import profile_imatrix as I


def string(value):
    raw = value.encode("ascii")
    return struct.pack("<Q", len(raw)) + raw


def fixture(path, mutate=None, importance=False):
    metadata = {"general.type": "imatrix", "imatrix.datasets": ["agent-corpus.txt"],
                "imatrix.chunk_count": 1, "imatrix.chunk_size": 4}
    tensors = []
    for layer in range(3, 45):
        for role in I.ROLES:
            values = [4.0 if 10 <= i < 18 else 0.0 for i in range(288)]
            tensors.append({"name": f"blk.{layer}.ffn_{role}_exps.weight.counts",
                            "shape": [1, 288], "type": 0, "values": values})
            if importance:
                tensors.append({"name": f"blk.{layer}.ffn_{role}_exps.weight.in_sum2",
                                "shape": [2048 if role == "down" else 4096, 288], "type": 0})
    if mutate:
        mutate(tensors, metadata)
    offset = 0
    for tensor in tensors:
        tensor["offset"] = offset
        offset += 4 * math.prod(tensor["shape"])
        offset = (offset + 31) // 32 * 32
    raw = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        raw += string(key)
        if isinstance(value, str):
            raw += struct.pack("<I", 8) + string(value)
        elif isinstance(value, list):
            raw += struct.pack("<IIQ", 9, 8, len(value)) + b"".join(string(v) for v in value)
        else:
            raw += struct.pack("<II", 4, value)
    for tensor in tensors:
        raw += string(tensor["name"]) + struct.pack("<I", len(tensor["shape"]))
        raw += struct.pack(f"<{len(tensor['shape'])}Q", *tensor["shape"])
        raw += struct.pack("<IQ", tensor["type"], tensor["offset"])
    start = (len(raw) + 31) // 32 * 32
    with path.open("wb") as fh:
        fh.write(raw)
        fh.truncate(start + offset)
        for tensor in tensors:
            if "values" in tensor:
                fh.seek(start + tensor["offset"])
                fh.write(struct.pack("<288f", *tensor["values"]))


def trace_record(ids=tuple(range(8)), layer=45, k=8, weights=(1.0,) * 8):
    return struct.pack("<ii8i8f", layer, k, *ids, *weights)


class ImatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.source = self.root / "counts.gguf"
        self.out = self.root / "profiles"

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranking_roundtrip_metrics_provenance_and_refuse_overwrite(self):
        fixture(self.source)
        report = I.build_profiles(self.source, self.out, slots=(32, 64))
        profile = report["profiles"][0]
        self.assertEqual(profile["mtp"]["reserved_slots"], 0)
        self.assertEqual(profile["full_pairs"], 42 * 288)
        ranked = P.read_profile(self.out / "trunk-45-full.strp", 45, 288, 3)
        self.assertEqual(ranked[:8], [(3, i) for i in range(10, 18)])
        self.assertEqual(ranked[8], (4, 10))
        metrics = profile["capacities"][0]
        self.assertEqual(metrics["trunk_hits"], 32 * 4)
        self.assertEqual(metrics["trunk_visits"], 42 * 32)
        self.assertEqual(metrics["cache_bytes"], 32 * I.EXPERT_BYTES)
        self.assertAlmostEqual(metrics["estimated_trunk_misses_per_row"], 336 * (1 - 32 / 336))
        self.assertEqual(metrics["zero_seed_same_total_capacity"]["trunk_hits"], 0)
        self.assertEqual(sum(metrics["per_layer_slots"].values()), 32)
        self.assertEqual(report["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report["source"]["datasets"], ["agent-corpus.txt"])
        self.assertIn("unknown", report["source"]["training_weight_fingerprint"])
        self.assertEqual(report, json.loads((self.out / "profiles.json").read_text()))
        text = (self.out / "trunk-45-full.txt").read_text().splitlines()
        self.assertEqual(tuple(map(int, text[0].split())), ranked[0])
        self.assertEqual(len(text), len(ranked))
        with self.assertRaises(FileExistsError):
            I.build_profiles(self.source, self.out, slots=(32,))

    def test_bad_counts(self):
        cases = [
            ("shape", lambda ts, md: ts[0].update(shape=[288, 1]), "shape"),
            ("type", lambda ts, md: ts[0].update(type=1), "F32"),
            ("missing", lambda ts, md: ts.__delitem__(slice(0, 3)), "completeness"),
            ("role", lambda ts, md: ts[1]["values"].__setitem__(0, 1), "role mismatch"),
            ("nan", lambda ts, md: ts[0]["values"].__setitem__(0, float("nan")), "finite"),
            ("fraction", lambda ts, md: ts[0]["values"].__setitem__(0, 0.5), "integers"),
            ("negative", lambda ts, md: ts[0]["values"].__setitem__(0, -1), "nonnegative"),
            ("sum", lambda ts, md: [t["values"].__setitem__(10, 3) for t in ts[:3]], "visit sum"),
            ("zero", lambda ts, md: [t.update(values=[0.0] * 288) for t in ts], "visit sum"),
            ("bound", lambda ts, md: ts[0]["values"].__setitem__(10, 5), "row bound"),
            ("layer", lambda ts, md: ts[0].update(name="blk.45.ffn_gate_exps.weight.counts"), "out of bounds"),
            ("duplicate", lambda ts, md: ts.append(dict(ts[0])), "duplicate tensor"),
            ("logical_duplicate", lambda ts, md: ts.append(dict(ts[0], name="blk.03.ffn_gate_exps.weight.counts")), "duplicate logical"),
            ("alignment", lambda ts, md: md.update({"general.alignment": 3}), "alignment"),
            ("kind", lambda ts, md: md.update({"general.type": "model"}), "general.type"),
            ("chunks", lambda ts, md: md.update({"imatrix.chunk_count": 0}), "positive integers"),
            ("datasets", lambda ts, md: md.update({"imatrix.datasets": []}), "datasets"),
        ]
        for name, mutate, message in cases:
            with self.subTest(name=name):
                fixture(self.source, mutate)
                with self.assertRaisesRegex(ValueError, message):
                    I.read_imatrix(self.source)
                self.assertFalse(self.out.exists())

    def test_bounds_truncated_header_and_stable_stat(self):
        fixture(self.source)
        with self.source.open("r+b") as fh:
            fh.truncate(self.source.stat().st_size - 32)
        with self.assertRaisesRegex(ValueError, "bounds"):
            I.read_imatrix(self.source)
        self.source.write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "malformed"):
            I.read_imatrix(self.source)
        fixture(self.source)
        before = I.stat_signature(self.source)
        with patch.object(I, "stat_signature", side_effect=[before, (*before[:3], 0, 0)]):
            with self.assertRaisesRegex(ValueError, "changed"):
                I.read_imatrix(self.source)

    def test_sparse_importance_header_only_validation(self):
        fixture(self.source, importance=True)
        # Avoid hashing sparse payloads; normal count-only fixtures exercise real hashing.
        with patch.object(I, "source_digest", return_value="test-only"):
            freq, _ = I.read_imatrix(self.source)
        self.assertEqual(sum(freq.values()), 42 * 32)
        fixture(self.source, lambda ts, md: ts[1].update(shape=[4095, 288]), importance=True)
        with patch.object(I, "source_digest", return_value="test-only"):
            with self.assertRaisesRegex(ValueError, "importance shape"):
                I.read_imatrix(self.source)

    def test_mtp_proxy_label_fixed_reserve_and_no_triple_count(self):
        fixture(self.source)
        report = I.build_profiles(self.source, self.out, slots=(40, 50), include_mtp=True)
        profile = report["profiles"][1]
        self.assertEqual(profile["mtp"]["status"], "placeholder")
        self.assertEqual(profile["mtp"]["counts_source"], "layer44proxy")
        self.assertEqual(profile["mtp"]["MTP"], "unobserved")
        rank = P.read_profile(self.out / profile["full_profile"], 46, 288, 3)
        self.assertEqual([p[0] for p in rank[:32]], [45] * 32)
        self.assertEqual(rank[32], (3, 10))
        for metrics in profile["capacities"]:
            self.assertEqual(metrics["per_layer_slots"]["45"], 32)
            self.assertEqual(metrics["trunk_visits"], 42 * 32)
            self.assertEqual(metrics["trunk_hits"], (metrics["slots"] - 32) * 4)
        zero_out = self.root / "zero-reserve"
        zero = I.build_profiles(self.source, zero_out, slots=(1,), include_mtp=True, mtp_slots=0)
        self.assertEqual(zero["profiles"][1]["capacities"][0]["per_layer_slots"]["45"], 0)
        with self.assertRaisesRegex(ValueError, "reserve"):
            I.build_profiles(self.source, self.root / "bad", slots=(32,), include_mtp=True)

    def test_measured_trace_counts_ids_not_weights_and_reserve(self):
        fixture(self.source)
        trace = self.root / "mtp.bin"
        trace.write_bytes(trace_record(tuple(range(20, 28)), weights=(0.0,) * 8) * 2 + trace_record())
        report = I.build_profiles(self.source, self.out, slots=(40,), include_mtp=True, mtp_trace=trace)
        profile = report["profiles"][1]
        self.assertEqual(profile["mtp"]["status"], "measured")
        self.assertEqual(profile["mtp"]["draft_rows"], 3)
        self.assertEqual(profile["mtp"]["sha256"], hashlib.sha256(trace.read_bytes()).hexdigest())
        rank = P.read_profile(self.out / profile["full_profile"], 46, 288, 3)
        self.assertEqual(rank[:8], [(45, i) for i in range(20, 28)])
        self.assertEqual(profile["capacities"][0]["trunk_hits"], 32)
        with self.assertRaisesRegex(ValueError, "requires --include-mtp"):
            I.build_profiles(self.source, self.root / "bad", mtp_trace=trace)

    def test_jsonl_measured_catchup_only(self):
        fixture(self.source)
        trace = self.root / "route45.jsonl"
        row = {"src": "accept", "kind": "round", "step": 0, "pos": 10,
               "ids": list(range(20, 28)), "w": [0.0] * 8}
        rows = [row, dict(row, pos=11), dict(row, kind="draft", step=1, ids=list(range(8))),
                dict(row, kind="prefill", ids=list(range(8))), dict(row, step=1, ids=list(range(8)))]
        trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report = I.build_profiles(self.source, self.out, slots=(40,), include_mtp=True, mtp_trace=trace)
        profile = report["profiles"][1]
        self.assertEqual(profile["mtp"]["measurement_scope"], "measuredteacherforcedcatchup-only")
        self.assertEqual(profile["mtp"]["catchup_rows"], 2)
        self.assertEqual(profile["mtp"]["draft_rows"], 0)
        self.assertEqual(profile["mtp"]["excluded_rows"], {"prefill": 1, "draft": 1, "round": 1})
        self.assertEqual(profile["mtp"]["expert_frequencies"][0], 0)
        self.assertEqual(profile["mtp"]["expert_frequencies"][20], 2)
        self.assertEqual(profile["mtp"]["sha256"], hashlib.sha256(trace.read_bytes()).hexdigest())
        rank = P.read_profile(self.out / profile["full_profile"], 46, 288, 3)
        self.assertEqual(rank[:8], [(45, i) for i in range(20, 28)])
        self.assertEqual(profile["capacities"][0]["trunk_hits"], 32)

    def test_bad_jsonl_records_including_excluded_rows(self):
        trace = self.root / "bad.jsonl"
        row = {"src": "accept", "kind": "round", "step": 0, "pos": 1,
               "ids": list(range(8)), "w": [1.0] * 8}
        cases = [("broken", "invalid"), ("[]", "object"),
                 (json.dumps(dict(row, kind="other")), "kind"),
                 (json.dumps(dict(row, layer=44)), "layer"),
                 (json.dumps(dict(row, ids=[0] * 8)), "unique"),
                 (json.dumps(dict(row, ids=[True] + list(range(1, 8)))), "integers"),
                 (json.dumps(dict(row, w=[float("nan")] * 8)), "finite"),
                 (json.dumps(dict(row, w=[10 ** 400] * 8)), "finite"),
                 (json.dumps(dict(row, kind="draft", step=1, ids=[0] * 8)), "unique"),
                 (json.dumps(dict(row, kind="draft", step=1)), "no round step0")]
        for raw, message in cases:
            with self.subTest(message=message):
                trace.write_text(raw + "\n")
                with self.assertRaisesRegex(ValueError, message):
                    I.read_mtp_trace(trace)

    def test_bad_mtp_records(self):
        trace = self.root / "bad.bin"
        cases = [(b"", "no draft"), (b"a", "truncated"),
                 (trace_record(layer=44), "layer=45"), (trace_record(k=7), "k=8"),
                 (trace_record(ids=(0,) * 8), "unique"),
                 (trace_record(ids=(-1, 1, 2, 3, 4, 5, 6, 7)), "range"),
                 (trace_record(weights=(float("nan"),) + (1.0,) * 7), "finite"),
                 (trace_record(weights=(-1.0,) + (1.0,) * 7), "nonnegative")]
        for raw, message in cases:
            with self.subTest(message=message):
                trace.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, message):
                    I.read_mtp_trace(trace)


if __name__ == "__main__":
    unittest.main()
