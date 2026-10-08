"""Extract selected real GGUF experts as a bounded IQ4 parity fixture.

Matching IQ4 payloads are preserved exactly, including calibrated final weights.
Other source types are requantized without calibration for kernel tests only.
References use the final IQ4 matrices, not the original BF16 or oracle matrices.
Files are little-endian; experts.bin is selected-expert gate|up|down blobs;
x.f32 is [T,4096], references are [selected expert,T,width], C order.
Requires NumPy and mainline libggml-base.so (ggml_type_traits ABI at 42d95816).
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import pathlib

import numpy as np
import pack_experts as E


class Traits(C.Structure):
    _fields_ = [("type_name", C.c_char_p), ("blck_size", C.c_int64),
                ("blck_size_interleave", C.c_int64), ("type_size", C.c_size_t),
                ("is_quantized", C.c_bool), ("to_float", C.c_void_p),
                ("from_float_ref", C.c_void_p)]


class GGML:
    def __init__(self, library):
        self.lib = C.CDLL(str(library))
        self.lib.ggml_get_type_traits.argtypes = [C.c_int]
        self.lib.ggml_get_type_traits.restype = C.POINTER(Traits)
        self.lib.ggml_quantize_requires_imatrix.argtypes = [C.c_int]
        self.lib.ggml_quantize_requires_imatrix.restype = C.c_bool
        self.lib.ggml_quantize_chunk.argtypes = [C.c_int, C.c_void_p, C.c_void_p,
                                               C.c_int64, C.c_int64, C.c_int64, C.c_void_p]
        self.lib.ggml_quantize_chunk.restype = C.c_size_t
        self.dequant = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int64)

    def traits(self, type_id):
        traits = self.lib.ggml_get_type_traits(type_id).contents
        block, size = E.G.BLOCK_GEOMETRY[E.G.GGML_TYPES[type_id]]
        if (traits.blck_size, traits.type_size) != (block, size):
            raise ValueError("ggml_type_traits ABI/geometry mismatch")
        if not traits.to_float:
            raise ValueError("GGML type has no dequantizer")
        return traits

    def decode(self, raw, type_id, rows, width):
        traits = self.traits(type_id)
        row_bytes = width // traits.blck_size * traits.type_size
        if len(raw) != rows * row_bytes:
            raise ValueError("wrong quantized matrix length")
        source = C.create_string_buffer(raw)
        values = np.empty((rows, width), dtype=np.float32)
        convert = self.dequant(traits.to_float)
        for row in range(rows):
            convert(C.addressof(source) + row * row_bytes, values.ctypes.data + row * width * 4, width)
        return values

    def quantize(self, values, type_id):
        traits = self.traits(type_id)
        if self.lib.ggml_quantize_requires_imatrix(type_id):
            raise ValueError("fixture quant requires an importance matrix")
        values = np.ascontiguousarray(values, dtype=np.float32)
        rows, width = values.shape
        output = C.create_string_buffer(rows * (width // traits.blck_size) * traits.type_size)
        written = self.lib.ggml_quantize_chunk(type_id, values.ctypes.data, C.addressof(output),
                                               0, rows, width, None)
        if written != C.sizeof(output):
            raise ValueError("GGML quantizer returned an unexpected byte count")
        return output.raw


def bf16_fixture_manifest(first, layer=3):
    """Fixture-only descriptor; do not relax the production IQ4 pack contract."""
    headers, where = E.read_headers(first)
    if headers[0].metadata.get("general.architecture") != "glm5-next":
        raise ValueError("BF16 fixture requires glm5-next source metadata")
    if not 3 <= layer <= 45:
        raise ValueError("BF16 fixture layer must be routed (3..45)")
    segments = []
    for role in E.ROLES:
        name = f"blk.{layer}.ffn_{role}_exps.weight"
        if name not in where:
            raise ValueError(f"BF16 fixture missing {name}")
        header, tensor = where[name]
        shape = [2048, 4096, 288] if role == "down" else [4096, 2048, 288]
        if tensor.shape != shape or tensor.type_name != "BF16":
            raise ValueError(f"{name}: BF16 fixture expected {shape} BF16, got {tensor.shape} {tensor.type_name}")
        expert_bytes = shape[0] * shape[1] * 2
        segments.append({"role": role, "tensor": name, "source_path": str(header.path),
                         "type_id": tensor.type_id, "type_name": tensor.type_name, "shape": shape,
                         "rows": shape[1], "tensor_offset": header.data_start + tensor.offset,
                         "expert_stride": expert_bytes, "expert_bytes": expert_bytes})
    return {"source": {"shard1": str(headers[0].path), "kind": "BF16",
                       "shards": [{"path": str(h.path), "file_bytes": h.path.stat().st_size,
                                   "data_start": h.data_start} for h in headers]},
            "layers": [{"layer": layer, "segments": segments}]}


def make_fixture(manifest, library, out, layer=3, ids=tuple(range(8)), tokens=4, x_path=None, seed=53):
    if not ids or len(ids) > 8 or len(set(ids)) != len(ids) or any(i < 0 or i >= 288 for i in ids):
        raise ValueError("select 1..8 distinct expert ids in 0..287")
    if not 1 <= tokens <= 8:
        raise ValueError("tokens must be 1..8 for a bounded fixture")
    source_layer = next((record for record in manifest["layers"] if record["layer"] == layer), None)
    if source_layer is None:
        raise ValueError("layer must be routed (3..45)")
    out = pathlib.Path(out)
    if out.exists():
        raise FileExistsError("fixture directory exists; use a fresh output path")
    ggml = GGML(library)
    if x_path:
        x = np.fromfile(x_path, dtype="<f4")
        if x.size != tokens * 4096:
            raise ValueError("x.f32 must contain tokens*4096 floats")
        x = x.reshape(tokens, 4096)
    else:
        x = np.random.default_rng(seed).normal(0, 0.1, (tokens, 4096)).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite fixture input")
    out.mkdir(parents=True)
    x.astype("<f4").tofile(out / "x.f32")
    meta = {"format": "strata-glm-expert-fixture", "version": 1, "quality_test_only": True,
            "source": manifest["source"], "library": str(pathlib.Path(library).absolute()),
            "layer": layer, "expert_ids": list(ids), "tokens": tokens, "hidden": 4096, "ff": 2048,
            "weights_preserved": all(s["type_id"] == (20 if s["role"] == "down" else 23)
                                     for s in source_layer["segments"]),
            "seed": seed, "reference": "F32 matmul of dequantized final IQ4 weights; clamp10 asymmetric",
            "blob_bytes": 13631488, "gate_up_bytes": 8912896, "down_bytes": 4718592,
            "segments": [{"role": "gate", "type_id": 23, "type_name": "IQ4_XS", "blob_offset": 0,
                          "row_bytes": 2176, "rows": 2048, "bytes": 4456448},
                         {"role": "up", "type_id": 23, "type_name": "IQ4_XS", "blob_offset": 4456448,
                          "row_bytes": 2176, "rows": 2048, "bytes": 4456448},
                         {"role": "down", "type_id": 20, "type_name": "IQ4_NL", "blob_offset": 8912896,
                          "row_bytes": 1152, "rows": 4096, "bytes": 4718592}],
            "reference_shapes": {"x.f32": [tokens, 4096], "gate.f32": [len(ids), tokens, 2048],
                                 "up.f32": [len(ids), tokens, 2048],
                                 "activation.f32": [len(ids), tokens, 2048],
                                 "output.f32": [len(ids), tokens, 4096]}}
    with (out / "experts.bin").open("xb") as blobs:
        refs = {name: (out / f"{name}.f32").open("xb") for name in ("gate", "up", "activation", "output")}
        try:
            for expert in ids:
                intermediate = {}
                for segment in source_layer["segments"]:
                    with open(segment["source_path"], "rb") as source:
                        source.seek(segment["tensor_offset"] + expert * segment["expert_stride"])
                        raw = source.read(segment["expert_bytes"])
                    target_id = 20 if segment["role"] == "down" else 23
                    if segment["type_id"] == target_id:
                        packed = raw  # Preserve calibrated final-GGUF IQ4 payloads byte-for-byte.
                    else:
                        values = ggml.decode(raw, segment["type_id"], segment["rows"], segment["shape"][0])
                        packed = ggml.quantize(values, target_id)
                        del values
                    del raw
                    blobs.write(packed)
                    final = ggml.decode(packed, target_id, segment["rows"], segment["shape"][0])
                    del packed
                    role = segment["role"]
                    if role == "down":
                        gate = np.minimum(intermediate["gate"], np.float32(10))
                        # Stable SiLU retains the gate's unclamped negative tail.
                        sigmoid = np.exp(-np.logaddexp(np.float32(0), -gate))
                        activation = gate * sigmoid * np.clip(intermediate["up"], -10, 10)
                        activation.astype("<f4").tofile(refs["activation"])
                        result = activation @ final.T
                        result.astype("<f4").tofile(refs["output"])
                    else:
                        intermediate[role] = x @ final.T
                        intermediate[role].astype("<f4").tofile(refs[role])
                    del final
                print(f"expert {expert}: IQ4 blob and same-weight F32 references", flush=True)
        finally:
            for fh in refs.values():
                fh.close()
    if (out / "experts.bin").stat().st_size != len(ids) * meta["blob_bytes"]:
        raise ValueError("fixture blob size mismatch")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.iterdir())
    if total >= 500 * 1024 * 1024:
        raise ValueError("fixture exceeds 500 MiB limit")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--bf16-source", action="store_true", help="fixture-only BF16 selected layer descriptor")
    ap.add_argument("--ggml-library", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--x-f32")
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()
    try:
        ids = tuple(int(i) for i in args.experts.split(","))
        manifest = (bf16_fixture_manifest(args.gguf, args.layer) if args.bf16_source
                    else E.build_manifest(args.gguf, True, require_v3=False))
        size = make_fixture(manifest, args.ggml_library, args.out, args.layer, ids,
                            args.tokens, args.x_f32, args.seed)
    except (ValueError, OSError) as exc:
        ap.exit(1, f"make_expert_fixture: {exc}\n")
    print(f"fixture: {size} bytes; selected payloads only, no full model copy")


if __name__ == "__main__":
    main()
