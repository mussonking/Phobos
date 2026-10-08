"""Targeted pack/profile tests. Full geometry fixtures are sparse, payload-free files."""
import collections
import json
import pathlib
import struct
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import gguf_reader as G
import make_profile as P
import pack_experts as E


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture(path, oracle=False, missing=None, mutate=None):
    metadata = {"general.architecture": "glm5-next", "glm5-next.block_count": 46,
                "glm5-next.nextn_predict_layers": 1, "glm5-next.embedding_length": 4096,
                "glm5-next.expert_feed_forward_length": 2048, "glm5-next.expert_count": 288,
                "glm5-next.expert_used_count": 8, "glm5-next.leading_dense_block_count": 3}
    tensors = []
    for layer in range(3, 46):
        tensors.append(G.TensorInfo(f"blk.{layer}.ffn_gate_inp.weight", [4096, 288], 0, "F32", 0))
        for role in E.ROLES:
            shape = [2048, 4096, 288] if role == "down" else [4096, 2048, 288]
            kind = ("Q5_K" if role == "down" else "Q4_K") if oracle else E.RECIPE[role]
            type_id = next(i for i, name in G.GGML_TYPES.items() if name == kind)
            tensors.append(G.TensorInfo(f"blk.{layer}.ffn_{role}_exps.weight", shape, type_id, kind, 0))
    tensors = [t for t in tensors if t.name != missing]
    if mutate:
        mutate(tensors, metadata)
    offset = 0
    for t in tensors:
        t.offset = offset
        offset += t.expected_bytes()
        offset = (offset + 31) // 32 * 32
    raw = struct.pack("<IIQQ", G.GGUF_MAGIC, 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        raw += string(key)
        raw += struct.pack("<I", 8 if isinstance(value, str) else 4)
        raw += string(value) if isinstance(value, str) else struct.pack("<I", value)
    for t in tensors:
        raw += string(t.name) + struct.pack("<I", len(t.shape))
        raw += struct.pack(f"<{len(t.shape)}Q", *t.shape) + struct.pack("<IQ", t.type_id, t.offset)
    start = (len(raw) + 31) // 32 * 32
    with path.open("wb") as fh:
        fh.write(raw)
        fh.truncate(start + offset)  # Logical model size only; never write model payloads.
    return tensors


class PackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.source = self.root / "model.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_custom_geometry_and_offsets(self):
        fixture(self.source)
        # Payload access would use a second open; each shard is opened once for header parsing.
        original = pathlib.Path.open
        opened = []
        def track(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)
        with patch.object(pathlib.Path, "open", track):
            manifest = E.build_manifest(self.source)
        self.assertEqual(opened, [self.source])
        self.assertEqual([record["layer"] for record in manifest["layers"]], list(range(3, 46)))
        layer = manifest["layers"][0]
        self.assertEqual(layer["blob_bytes"], 13631488)
        self.assertEqual(layer["gate_up_bytes"], 8912896)
        self.assertEqual(layer["down_bytes"], 4718592)
        self.assertTrue(all(record["blob_bytes"] == record["gate_up_bytes"] + record["down_bytes"]
                            for record in manifest["layers"]))
        self.assertEqual([s["row_bytes"] for s in layer["segments"]], [2176, 2176, 1152])
        self.assertEqual([s["blob_offset"] for s in layer["segments"]], [0, 4456448, 8912896])
        self.assertEqual(manifest["total_bytes"], 43 * 288 * 13631488)
        E.write_manifests(manifest, self.root / "pack")
        self.assertFalse((self.root / "pack/experts.bin").exists())
        rows = (self.root / "pack/native_experts.txt").read_text().splitlines()[2:]
        self.assertEqual(len(rows), 43)
        self.assertEqual(list(map(int, rows[0].split()[:5])), [3, 23, 20, 0, 13631488])
        self.assertEqual(json.loads((self.root / "pack/native_experts.json").read_text()), manifest)
        with self.assertRaises(FileExistsError):
            E.write_manifests(manifest, self.root / "pack")

    def test_oracle_opt_in(self):
        fixture(self.source, oracle=True)
        with self.assertRaisesRegex(ValueError, "expected.*IQ4"):
            E.build_manifest(self.source)
        manifest = E.build_manifest(self.source, True)
        self.assertEqual(manifest["layers"][0]["segments"][2]["type_name"], "Q5_K")

    def test_missing_mtp(self):
        fixture(self.source, missing="blk.45.ffn_down_exps.weight")
        with self.assertRaisesRegex(ValueError, "completeness.*blk.45"):
            E.build_manifest(self.source)

    def test_wrong_metadata(self):
        fixture(self.source, mutate=lambda ts, md: md.update({"glm5-next.block_count": 45}))
        with self.assertRaisesRegex(ValueError, "block_count"):
            E.build_manifest(self.source)

    def test_wrong_expert_shape(self):
        fixture(self.source, mutate=lambda ts, md: setattr(ts[1], "shape", [4096, 2048, 287]))
        with self.assertRaisesRegex(ValueError, "expected.*288"):
            E.build_manifest(self.source)

    def test_wrong_router_shape(self):
        fixture(self.source, mutate=lambda ts, md: setattr(ts[0], "shape", [4096, 287]))
        with self.assertRaisesRegex(ValueError, "router"):
            E.build_manifest(self.source)

    def test_gate_up_mismatch(self):
        def change(ts, md):
            ts[2].type_id, ts[2].type_name = 13, "Q5_K"
        fixture(self.source, oracle=True, mutate=change)
        with self.assertRaisesRegex(ValueError, "matching gate/up"):
            E.build_manifest(self.source, True)

    def test_truncated_tensor(self):
        fixture(self.source)
        with self.source.open("r+b") as fh:
            fh.truncate(self.source.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "truncated tensor"):
            E.build_manifest(self.source)

    def test_duplicate_tensor(self):
        fixture(self.source, mutate=lambda ts, md: ts.append(G.TensorInfo(ts[1].name, ts[1].shape, 23, "IQ4_XS", 0)))
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            E.build_manifest(self.source)

    def test_missing_shard(self):
        first = self.root / "model-00001-of-00002.gguf"
        fixture(first)
        with self.assertRaises(FileNotFoundError):
            E.build_manifest(first)

    def test_shards_preserve_source_and_v3_basenames(self):
        first = self.root / "model-00001-of-00002.gguf"
        second = self.root / "model-00002-of-00002.gguf"
        fixture(first, mutate=lambda ts, md: ts.__setitem__(slice(None), [
            t for t in ts if int(t.name.split(".")[1]) < 24]))
        fixture(second, mutate=lambda ts, md: ts.__setitem__(slice(None), [
            t for t in ts if int(t.name.split(".")[1]) >= 24]))
        manifest = E.build_manifest(first)
        self.assertEqual(manifest["layers"][-1]["segments"][0]["source_path"], str(second))
        E.write_manifests(manifest, self.root / "pack")
        rows = (self.root / "pack/native_experts.txt").read_text().splitlines()
        self.assertEqual(rows[-1].split()[-1], second.name)
        self.assertEqual(len(rows[2].split()), 8)

    def test_split_layer_is_rejected(self):
        first = self.root / "model-00001-of-00002.gguf"
        second = self.root / "model-00002-of-00002.gguf"
        name = "blk.45.ffn_down_exps.weight"
        fixture(first, missing=name)
        fixture(second, mutate=lambda ts, md: ts.__setitem__(slice(None), [t for t in ts if t.name == name]))
        with self.assertRaisesRegex(ValueError, "split across shards"):
            E.build_manifest(first)
        manifest = E.build_manifest(first, require_v3=False)
        E.write_manifests(manifest, self.root / "json-pack")
        self.assertTrue((self.root / "json-pack/native_experts.json").exists())
        self.assertFalse((self.root / "json-pack/native_experts.txt").exists())
        self.assertEqual(manifest["layers"][-1]["segments"][-1]["source_path"], str(second))

    def test_overlap_and_alignment_rejected(self):
        fixture(self.source)
        header = G.GGUFFile(self.source)
        header.tensors[1].offset = 0
        with patch.object(G, "GGUFFile", return_value=header):
            with self.assertRaisesRegex(ValueError, "overlapping"):
                E.read_headers(self.source)
        header.tensors[1].offset = 1
        with patch.object(G, "GGUFFile", return_value=header):
            with self.assertRaisesRegex(ValueError, "misaligned"):
                E.read_headers(self.source)

    def test_malformed_header(self):
        self.source.write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "malformed GGUF header"):
            E.read_headers(self.source)

    def test_stream_gather_order_bounded(self):
        self.source.write_bytes(b"aabbCCDDeeff")
        segments = [{"source_path": str(self.source), "tensor_offset": off,
                     "expert_stride": 2, "expert_bytes": 2, "tensor": role}
                    for off, role in zip((0, 4, 8), E.ROLES)]
        manifest = {"source": {"shards": [{"path": str(self.source)}]}, "geometry": {"n_expert": 2},
                    "total_bytes": 12, "layers": [{"arena_offset": 0, "segments": segments}]}
        output = self.root / "experts.bin"
        E.stream_experts(manifest, output, chunk_bytes=1)
        self.assertEqual(output.read_bytes(), b"aaCCeebbDDff")
        with self.assertRaises(FileExistsError):
            E.stream_experts(manifest, output)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "profile.bin"

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_frequency_full_profile_and_wire_format(self):
        ranked = P.rank_pairs(collections.Counter())
        self.assertEqual(len(ranked), 43 * 288)
        self.assertEqual(ranked[:2], [(3, 0), (4, 0)])
        self.assertIn((45, 287), ranked)
        P.write_profile(self.path, ranked)
        self.assertEqual(P.read_profile(self.path), ranked)
        raw = self.path.read_bytes()
        self.assertEqual(raw[:24], b"STRP" + struct.pack("<5I", 1, 46, 288, 12384, 12384))
        dense = struct.unpack_from("<864i", raw, 24 + 4 * len(ranked))
        self.assertEqual(set(dense), {-1})

    def test_dynamic_geometry_and_ranking(self):
        freq = collections.Counter({(4, 2): 9, (3, 1): 0})
        ranked = P.rank_pairs(freq, [(3, 0)], 5, 3, 3)
        self.assertEqual(ranked[:3], [(3, 0), (4, 2), (3, 1)])
        P.write_profile(self.path, ranked, 5, 3, 3)
        self.assertEqual(P.read_profile(self.path, 5, 3, 3), ranked)

    def test_invalid_profile_pairs_and_lengths(self):
        for pairs in [[(0, 0)], [(46, 0)], [(3, 288)], [(3, 0), (3, 0)]]:
            with self.assertRaises(ValueError):
                P.write_profile(self.path, pairs)
        P.write_profile(self.path, [(3, 0)])
        raw = self.path.read_bytes()
        for bad in [raw[:-1], raw + b"x", raw[:4] + struct.pack("<I", 2) + raw[8:]]:
            self.path.write_bytes(bad)
            with self.assertRaises(ValueError):
                P.read_profile(self.path)

    def test_trace_validation_and_zero_weights(self):
        raw = struct.pack("<4i2f", 45, 2, 0, 287, 0., 1.)
        self.path.write_bytes(raw)
        self.assertEqual(P.read_trace(self.path), {(45, 0): 1, (45, 287): 1})
        for bad in [raw[:-1], raw + b"x", struct.pack("<ii", 3, -1),
                    struct.pack("<3if", 0, 1, 0, 1.),
                    struct.pack("<4i2f", 3, 2, 0, 0, 1., 1.),
                    struct.pack("<3if", 3, 1, 0, float("nan"))]:
            self.path.write_bytes(bad)
            with self.assertRaises(ValueError):
                P.read_trace(self.path)


if __name__ == "__main__":
    unittest.main()
