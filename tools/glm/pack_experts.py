"""Header-only GLM5-next native expert manifest; payload copying is opt-in.

native_experts.txt uses Strata v3 syntax. Layers 0..2 are intentionally absent:
a GLM consumer must treat these dense layers as having no routed expert blobs.
The source GGUF has three separate tensors, not contiguous expert blobs. Gather
three slices into a cache slot in gate | up | down order; no repacked model is
needed. native_experts.json makes that gather contract explicit.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import struct
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import gguf_reader as G

ROLES = ("gate", "up", "down")
GEOMETRY = {"n_layers": 46, "n_trunk_layers": 45, "n_expert": 288,
            "n_expert_used": 8, "hidden": 4096, "ff": 2048,
            "first_moe_layer": 3, "mtp_layer": 45}
RECIPE = {"gate": "IQ4_XS", "up": "IQ4_XS", "down": "IQ4_NL"}
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def shard_paths(first):
    first = pathlib.Path(first).absolute()  # Keep HF symlink shard names.
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not match:
        return [first]
    count = int(match[2])
    if int(match[1]) != 1 or count < 1:
        raise ValueError("--gguf must name shard 1")
    return [first.with_name(first.name[:match.start()] + f"-{i:05d}-of-{count:05d}.gguf")
            for i in range(1, count + 1)]


def read_headers(first):
    """Validate tensor extents without mapping or reading any tensor payload."""
    headers, where = [], {}
    for path in shard_paths(first):
        try:
            header = G.GGUFFile(path)
        except (struct.error, KeyError, OverflowError) as exc:
            raise ValueError(f"{path}: malformed GGUF header") from exc
        size = path.stat().st_size
        if header.alignment <= 0 or header.alignment & (header.alignment - 1):
            raise ValueError(f"{path}: invalid GGUF alignment")
        if header.data_start > size:
            raise ValueError(f"{path}: truncated GGUF data start")
        extents = []
        for tensor in header.tensors:
            if not tensor.shape or any(d <= 0 for d in tensor.shape):
                raise ValueError(f"{path}: invalid dimensions for {tensor.name}")
            geom = G.BLOCK_GEOMETRY.get(tensor.type_name)
            nbytes = tensor.expected_bytes()
            if geom is None or tensor.shape[0] % geom[0] or nbytes is None:
                raise ValueError(f"{path}: unsupported row geometry for {tensor.name}")
            start = header.data_start + tensor.offset
            if tensor.offset % header.alignment or start + nbytes > size:
                raise ValueError(f"{path}: misaligned or truncated tensor {tensor.name}")
            if tensor.name in where:
                raise ValueError(f"{path}: duplicate tensor {tensor.name}")
            where[tensor.name] = (header, tensor)
            extents.append((start, start + nbytes, tensor.name))
        extents.sort()
        for prev, current in zip(extents, extents[1:]):
            if prev[1] > current[0]:
                raise ValueError(f"{path}: overlapping tensors {prev[2]} and {current[2]}")
        headers.append(header)
    return headers, where


def build_manifest(first, allow_oracle_types=False, require_v3=True):
    headers, where = read_headers(first)
    required = {"general.architecture": "glm5-next", "glm5-next.block_count": 46,
                "glm5-next.nextn_predict_layers": 1, "glm5-next.embedding_length": 4096,
                "glm5-next.expert_feed_forward_length": 2048, "glm5-next.expert_count": 288,
                "glm5-next.expert_used_count": 8, "glm5-next.leading_dense_block_count": 3}
    for key, expected in required.items():
        if headers[0].metadata.get(key) != expected:
            raise ValueError(f"{key}: expected {expected!r}, got {headers[0].metadata.get(key)!r}")
        for header in headers[1:]:
            if key in header.metadata and header.metadata[key] != expected:
                raise ValueError(f"{header.path}: inconsistent {key}")
    expected_names = {f"blk.{layer}.ffn_{role}_exps.weight"
                      for layer in range(3, 46) for role in ROLES}
    actual_names = {name for name in where if name.endswith("_exps.weight")}
    if actual_names != expected_names:
        raise ValueError(f"expert tensor completeness: missing {sorted(expected_names - actual_names)}, "
                         f"unexpected {sorted(actual_names - expected_names)}")
    layers, arena = [], 0
    for layer in range(3, 46):
        router = where.get(f"blk.{layer}.ffn_gate_inp.weight")
        if router is None or router[1].shape != [4096, 288]:
            raise ValueError(f"layer {layer}: missing or invalid router [4096,288]")
        segments, blob_at = [], 0
        for role in ROLES:
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            header, tensor = where[name]
            shape = [2048, 4096, 288] if role == "down" else [4096, 2048, 288]
            if tensor.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {tensor.shape}")
            allowed = {RECIPE[role]}
            if allow_oracle_types:
                allowed.update(("Q4_K", "Q5_K", "Q6_K"))
            if tensor.type_name not in allowed:
                raise ValueError(f"{name}: expected {sorted(allowed)}, got {tensor.type_name}")
            block, size = G.BLOCK_GEOMETRY[tensor.type_name]
            row_bytes = shape[0] // block * size
            expert_bytes = row_bytes * shape[1]
            segments.append({"role": role, "tensor": name, "source_path": str(header.path),
                             "shard": header.path.name, "type_id": tensor.type_id,
                             "type_name": tensor.type_name, "shape": shape,
                             "tensor_offset": header.data_start + tensor.offset,
                             "tensor_bytes": tensor.expected_bytes(), "row_bytes": row_bytes,
                             "rows": shape[1], "expert_stride": expert_bytes,
                             "expert_bytes": expert_bytes, "blob_offset": blob_at})
            blob_at += expert_bytes
        if segments[0]["type_id"] != segments[1]["type_id"]:
            raise ValueError(f"layer {layer}: v3 requires matching gate/up types")
        split = len({s["source_path"] for s in segments}) != 1
        if split and require_v3:
            raise ValueError(f"layer {layer}: v3 cannot represent gate/up/down split across shards")
        if any(c.isspace() for s in segments for c in s["shard"]):
            raise ValueError("shard basenames cannot contain whitespace")
        layers.append({"layer": layer, "arena_offset": arena, "blob_bytes": blob_at,
                       "gate_up_bytes": segments[0]["expert_bytes"] + segments[1]["expert_bytes"],
                       "down_bytes": segments[2]["expert_bytes"], "segments": segments})
        arena += blob_at * 288
    return {"format": "strata-glm-native-experts", "version": 3, "geometry": dict(GEOMETRY),
            "source": {"shard1": str(headers[0].path), "shards": [
                {"path": str(h.path), "name": h.path.name, "file_bytes": h.path.stat().st_size,
                 "data_start": h.data_start} for h in headers]},
            "oracle_types_allowed": allow_oracle_types, "v3_representable": require_v3, "blob_order": list(ROLES),
            "source_expert_offset": "tensor_offset + expert_id * expert_stride",
            "cache_blob_offset": "arena_offset + expert_id * blob_bytes",
            "total_bytes": arena, "layers": layers}


def write_manifests(manifest, out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / "native_experts.txt", out / "native_experts.json", out / "experts.bin"]
    if any(p.exists() for p in targets):
        raise FileExistsError("refusing to replace existing expert pack files; use a fresh --out directory")
    if not manifest["v3_representable"]:
        (out / "native_experts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return
    with (out / "native_experts.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
                 f"(n_expert 288, total {manifest['total_bytes']}; absolute offsets in "
                 f"{pathlib.Path(manifest['source']['shard1']).name}, or in the named shard beside it)\n")
        fh.write("# GLM5-next: dense layers 0..2 absent; routed layers 3..45 include MTP45; hidden4096 ff2048\n")
        for layer in manifest["layers"]:
            parts = layer["segments"]
            fields = [layer["layer"], parts[0]["type_id"], parts[2]["type_id"],
                      layer["arena_offset"], layer["blob_bytes"],
                      *[s["tensor_offset"] for s in parts]]
            shard = "" if parts[0]["source_path"] == manifest["source"]["shard1"] else " " + parts[0]["shard"]
            fh.write(" ".join(map(str, fields)) + shard + "\n")
    (out / "native_experts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def stream_experts(manifest, path, chunk_bytes=1024 * 1024):
    """Bounded memory gather; never allocate a layer-sized or model-sized buffer."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    path = pathlib.Path(path)
    with contextlib.ExitStack() as stack:
        inputs = {s["path"]: stack.enter_context(open(s["path"], "rb"))
                  for s in manifest["source"]["shards"]}
        output = stack.enter_context(path.open("xb"))
        for layer in manifest["layers"]:
            if output.tell() != layer["arena_offset"]:
                raise ValueError("noncontiguous expert arena")
            for expert in range(manifest["geometry"]["n_expert"]):
                for segment in layer["segments"]:
                    source = inputs[segment["source_path"]]
                    source.seek(segment["tensor_offset"] + expert * segment["expert_stride"])
                    remaining = segment["expert_bytes"]
                    while remaining:
                        raw = source.read(min(chunk_bytes, remaining))
                        if not raw:
                            raise ValueError(f"source truncated while gathering {segment['tensor']}")
                        output.write(raw)
                        remaining -= len(raw)
        if output.tell() != manifest["total_bytes"]:
            raise ValueError("expert arena byte count mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, help="merged GGUF or shard 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-oracle-types", action="store_true", help="also accept Q4_K/Q5_K/Q6_K experts")
    ap.add_argument("--json-only", action="store_true",
                    help="allow oracle layers split across shards; writes only native_experts.json")
    ap.add_argument("--experts-bin", action="store_true", help="explicitly stream a full duplicate expert arena")
    args = ap.parse_args()
    try:
        manifest = build_manifest(args.gguf, args.allow_oracle_types, not args.json_only)
        write_manifests(manifest, args.out)
        if args.experts_bin:
            stream_experts(manifest, pathlib.Path(args.out) / "experts.bin")
    except (ValueError, OSError) as exc:
        ap.exit(1, f"pack_experts: {exc}\n")
    print(f"43 routed layers including MTP45; {manifest['total_bytes']} expert bytes; "
          + ("experts.bin streamed" if args.experts_bin else "header-only, no tensor payload copied"))


if __name__ == "__main__":
    main()
