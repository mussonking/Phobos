#!/usr/bin/env python3
"""Plan and verify the GLM native IQ4 quant, using GGUF headers only.

Non-experts mirror the oracle's exact tensor types; routed gate/up become
IQ4_XS and routed down becomes IQ4_NL. Every override is an anchored regex.
A real quant must first pass verify-dry-run, then verify-output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pathlib
import re
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile

EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
LINE = re.compile(r"\[\s*\d+/\s*\d+\]\s+(\S+)\s+-\s+\[[^\]]+\], type =\s*(\S+),\s+size =[^\n]*")
TARGET = re.compile(r"->[^\n]*\(([^)]+)\)\s*$")


def read_model(first: pathlib.Path, check_payload: bool = True):
    match = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", first.name)
    paths = ([first.with_name(f"{match[1]}-{i:05d}-of-{int(match[2]):05d}.gguf")
              for i in range(1, int(match[2]) + 1)] if match else [first])
    tensors = {}
    shards = []
    for path in paths:
        g = GGUFFile(path)
        size = path.stat().st_size
        for t in g.tensors:
            if t.name in tensors:
                raise ValueError(f"duplicate tensor {t.name}")
            nbytes = t.expected_bytes()
            if nbytes is None or t.shape[0] % BLOCK_GEOMETRY[t.type_name][0]:
                raise ValueError(f"invalid row geometry {t.name}")
            if check_payload and g.data_start + t.offset + nbytes > size:
                raise ValueError(f"truncated tensor {path.name}:{t.name}")
            tensors[t.name] = t
        shards.append({"path": str(path.resolve()), "bytes": size,
                       "data_start": g.data_start, "tensors": len(g.tensors)})
    metadata = GGUFFile(first).metadata
    if metadata.get("general.architecture") != "glm5-next":
        raise ValueError("expected glm5-next architecture")
    for key, val in {"block_count": 46, "embedding_length": 4096,
                     "expert_count": 288, "expert_used_count": 8,
                     "expert_feed_forward_length": 2048,
                     "nextn_predict_layers": 1}.items():
        if metadata.get("glm5-next." + key) != val:
            raise ValueError(f"unexpected geometry {key}: {metadata.get('glm5-next.' + key)}")
    return tensors, shards


def wanted_type(tensor):
    m = EXPERT.match(tensor.name)
    if m:
        return "IQ4_NL" if m[2] == "down" else "IQ4_XS"
    if tensor.type_name not in {"F32", "Q8_0"}:
        raise ValueError(f"oracle non-expert {tensor.name} is {tensor.type_name}, not F32/Q8_0")
    return tensor.type_name


def plan(args):
    oracle, _ = read_model(args.oracle)
    source, source_shards = read_model(args.source)
    if source.keys() != oracle.keys():
        raise ValueError(f"source/oracle tensor mismatch: source-only={sorted(source.keys()-oracle.keys())}, "
                         f"oracle-only={sorted(oracle.keys()-source.keys())}")
    records = []
    for name, t in sorted(oracle.items()):
        src = source[name]
        if src.shape != t.shape:
            raise ValueError(f"source/oracle shape mismatch {name}")
        target = wanted_type(t)
        block, nbytes = BLOCK_GEOMETRY[target]
        if t.shape[0] % block:
            raise ValueError(f"{name}: inner dimension not divisible by {block}")
        records.append({"name": name, "shape": t.shape, "source_type": src.type_name,
                        "oracle_type": t.type_name, "target_type": target,
                        "bytes": t.elements // block * nbytes})
    mainline_exclusions = None
    if args.accept_mainline_bf16:
        dry_text = args.accept_mainline_bf16.read_text(errors="replace")
        if not re.search(r"quant size\s*=", dry_text):
            raise ValueError("mainline acceptance requires a completed dry-run summary")
        dry_types = parse_dry_types(dry_text)
        if dry_types.keys() != source.keys():
            raise ValueError("mainline acceptance requires a complete dry-run tensor set")
        extra_trunk = 0
        extra_total = 0
        accepted = []
        for r in records:
            actual = dry_types[r["name"]]
            if actual == r["target_type"]:
                continue
            if actual != "BF16" or r["source_type"] != "BF16" or r["target_type"] != "Q8_0":
                raise ValueError(f"cannot accept mainline change {r['name']}: {r['target_type']} -> {actual}")
            elems = source[r["name"]].elements
            delta = elems * 2 - r["bytes"]
            extra_total += delta
            layer = re.match(r"blk\.(\d+)\.", r["name"])
            if not layer or int(layer[1]) < 45:
                extra_trunk += delta
            r["target_type"] = "BF16"
            r["bytes"] = elems * 2
            accepted.append(r["name"])
        if extra_trunk >= 500_000_000:
            raise ValueError(f"mainline BF16 extra per trunk token {extra_trunk} exceeds 0.5 GB decision threshold")
        mainline_exclusions = {"tensors": accepted, "extra_stored_bytes": extra_total,
                               "extra_trunk_bytes_per_token": extra_trunk,
                               "trunk_threshold_bytes": 500_000_000,
                               "dry_log_sha256": hashlib.sha256(dry_text.encode()).hexdigest()}
        sys.stdout.write(f"Accepted {len(accepted)} mainline BF16 exclusions: extra stored={extra_total} B,"
                         f"extra trunk per token={extra_trunk} B (<0.5 GB)\n")
    expert_records = [r for r in records if EXPERT.match(r["name"])]
    expected = {f"blk.{il}.ffn_{part}_exps.weight"
                for il in range(3, 46) for part in ("gate", "up", "down")}
    if {r["name"] for r in expert_records} != expected:
        raise ValueError("missing/extra expert layers (must preserve block 45 MTP)")
    for r in expert_records:
        part = EXPERT.match(r["name"])[2]
        shape = [2048, 4096, 288] if part == "down" else [4096, 2048, 288]
        if r["shape"] != shape:
            raise ValueError(f"{r['name']}: unexpected expert shape")
    payload = sum(r["bytes"] for r in records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overrides = args.out_dir / "tensor-types.txt"
    rules = "\n".join(f"^{re.escape(r['name'])}$={r['target_type'].lower()}" for r in records) + "\n"
    overrides.write_text(rules)
    argv = [str(args.quantizer), "--imatrix", str(args.imatrix), "--pure",
            "--tensor-type-file", str(overrides.resolve()),
            "--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0",
            "--max-buffer-size", "512", str(args.source.resolve()),
            str(args.output.resolve()), "Q8_0", str(args.threads)]
    recipe = {"format": "glm-iq4-recipe-v1", "source": str(args.source.resolve()),
              "oracle": str(args.oracle.resolve()), "imatrix": str(args.imatrix.resolve()),
              "output": str(args.output.resolve()), "source_shards": source_shards,
              "expected_payload_bytes": payload, "tensor_count": len(records),
              "expert_tensor_count": len(expert_records),
              "mainline_exclusions": mainline_exclusions,
              "type_counts": dict(Counter(r["target_type"] for r in records)),
              "overrides_sha256": hashlib.sha256(rules.encode()).hexdigest(),
              "argv": argv, "tensors": records}
    (args.out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    # stdout is informational only; the verification consumes the immutable JSON.
    print(json.dumps({k: recipe[k] for k in ("tensor_count", "expert_tensor_count",
                                           "type_counts", "expected_payload_bytes")}, indent=2))
    print("DRY_RUN", shlex.join([argv[0], "--dry-run"] + argv[1:]))
    print("REAL_RUN", shlex.join(argv))


def load_recipe(path):
    recipe = json.loads(path.read_text())
    if recipe.get("format") != "glm-iq4-recipe-v1":
        raise ValueError("unknown recipe format")
    argv = recipe["argv"]
    types_path = pathlib.Path(argv[argv.index("--tensor-type-file") + 1])
    if hashlib.sha256(types_path.read_bytes()).hexdigest() != recipe["overrides_sha256"]:
        raise ValueError("tensor overrides changed after planning")
    return recipe


def parse_dry_types(text):
    actual = {}
    for match in LINE.finditer(text):
        name, original = match[1], match[2].upper()
        target = TARGET.search(match[0])
        if name in actual:
            raise ValueError(f"duplicate dry-run tensor {name}")
        actual[name] = target[1].upper() if target else original
    return actual


def verify_dry(args):
    recipe = load_recipe(args.recipe)
    text = args.log.read_text(errors="replace")
    actual = parse_dry_types(text)
    expected = {r["name"]: r["target_type"] for r in recipe["tensors"]}
    mismatches = [f"{n}: wanted {t}, got {actual.get(n, 'MISSING')}"
                  for n, t in expected.items() if actual.get(n) != t]
    mismatches += [f"unexpected dry-run tensor {n}" for n in actual.keys()-expected.keys()]
    if mismatches:
        raise ValueError("dry-run type verification FAILED:\n" + "\n".join(mismatches))
    if not re.search(r"quant size\s*=", text):
        raise ValueError("dry-run incomplete (missing quant size summary)")
    print(f"PASS dry-run: {len(expected)} tensors, every type matches, MTP retained")


def verify_output(args):
    recipe = load_recipe(args.recipe)
    actual, shards = read_model(args.output)
    if len(shards) != 1:
        raise ValueError("output must be one merged GGUF")
    expected = {r["name"]: r for r in recipe["tensors"]}
    if actual.keys() != expected.keys():
        raise ValueError("output tensor set differs from recipe")
    failures = []
    for n, r in expected.items():
        t = actual[n]
        if t.type_name != r["target_type"] or t.shape != r["shape"]:
            failures.append(f"{n}: got {t.type_name}/{t.shape}, wanted {r['target_type']}/{r['shape']}")
    if failures:
        raise ValueError("output verification FAILED:\n" + "\n".join(failures))
    print(f"PASS output: {len(actual)} tensors; types, shapes, file bounds, single merged GGUF and MTP verified")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    for option in ("source", "oracle", "imatrix", "quantizer", "output", "out-dir"):
        s.add_argument("--" + option, type=pathlib.Path, required=True)
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--accept-mainline-bf16", type=pathlib.Path,
                   help="verified dry-run log; accept BF16 exclusions only if trunk overhead <0.5 GB")
    s = sub.add_parser("verify-dry-run")
    s.add_argument("--recipe", type=pathlib.Path, required=True)
    s.add_argument("--log", type=pathlib.Path, required=True)
    s = sub.add_parser("verify-output")
    s.add_argument("--recipe", type=pathlib.Path, required=True)
    s.add_argument("--output", type=pathlib.Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        {"plan": plan, "verify-dry-run": verify_dry,
         "verify-output": verify_output}[args.cmd](args)
    except (ValueError, OSError, KeyError) as exc:
        sys.exit(str(exc))
