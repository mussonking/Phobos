"""Build frequency-optimal STRP cache profiles from GLM imatrix visit counts.

Coverage is in-sample routing coverage, not throughput or out-of-corpus hit rate.
Importance magnitudes are never used. Gate/up/down counts describe the same
visits and are checked for equality, then counted once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import sys
from collections import Counter

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile
import make_profile as P

LAYERS, EXPERTS, FIRST_MOE, TOP_K = 45, 288, 3, 8
EXPERT_BYTES = 13 * 1024 * 1024
ROLES = ("gate", "up", "down")
COUNT_NAME = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\.counts$")


def stat_signature(path):
    st = pathlib.Path(path).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def source_digest(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fh:
        while block := fh.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_imatrix(path):
    path = pathlib.Path(path)
    before = stat_signature(path)
    try:
        gguf = GGUFFile(path)
    except (struct.error, KeyError, OverflowError) as exc:
        raise ValueError(f"malformed GGUF header: {exc}") from exc
    md = gguf.metadata
    if gguf.alignment <= 0 or gguf.alignment & (gguf.alignment - 1):
        raise ValueError("GGUF alignment must be a positive power of two")
    if md.get("general.type") != "imatrix":
        raise ValueError("general.type must be imatrix")
    chunks, chunk_size = md.get("imatrix.chunk_count"), md.get("imatrix.chunk_size")
    if any(type(v) is not int or v <= 0 for v in (chunks, chunk_size)):
        raise ValueError("imatrix.chunk_count and imatrix.chunk_size must be positive integers")
    datasets = md.get("imatrix.datasets")
    if not isinstance(datasets, list) or not datasets or any(not isinstance(v, str) or not v for v in datasets):
        raise ValueError("imatrix.datasets must be a nonempty string array")
    expected_rows = chunks * chunk_size
    expected_visits = TOP_K * expected_rows
    names, ranges, counts = {}, [], {}
    for tensor in gguf.tensors:
        if tensor.name in names:
            raise ValueError(f"duplicate tensor {tensor.name}")
        names[tensor.name] = tensor
        size = tensor.expected_bytes()
        if size is None or any(d <= 0 for d in tensor.shape):
            raise ValueError(f"unsupported tensor type/shape: {tensor.name}")
        start = gguf.data_start + tensor.offset
        if tensor.offset % gguf.alignment or start < gguf.data_start or start + size > before[2]:
            raise ValueError(f"tensor bounds/alignment invalid: {tensor.name}")
        ranges.append((start, start + size, tensor.name))
        match = COUNT_NAME.fullmatch(tensor.name)
        if match:
            layer, role = int(match[1]), match[2]
            if not FIRST_MOE <= layer < LAYERS:
                raise ValueError(f"count layer out of bounds: {tensor.name}")
            if tensor.type_id != 0 or tensor.shape != [1, EXPERTS]:
                raise ValueError(f"expected F32 shape [1, {EXPERTS}]: {tensor.name}")
            if (layer, role) in counts:
                raise ValueError(f"duplicate logical count tensor: {tensor.name}")
            counts[layer, role] = tensor
    ranges.sort()
    for left, right in zip(ranges, ranges[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping tensors: {left[2]} and {right[2]}")
    expected = {(layer, role) for layer in range(FIRST_MOE, LAYERS) for role in ROLES}
    if set(counts) != expected:
        raise ValueError(f"count completeness: missing {sorted(expected - set(counts))}")
    freq = Counter()
    with path.open("rb") as fh:
        for layer in range(FIRST_MOE, LAYERS):
            roles = []
            for role in ROLES:
                tensor = counts[layer, role]
                importance = names.get(tensor.name.removesuffix(".counts") + ".in_sum2")
                if importance is not None:
                    shape = [2048 if role == "down" else 4096, EXPERTS]
                    if importance.type_id != 0 or importance.shape != shape:
                        raise ValueError(f"expected F32 importance shape {shape}: {importance.name}")
                fh.seek(gguf.data_start + tensor.offset)
                raw = fh.read(EXPERTS * 4)
                if len(raw) != EXPERTS * 4:
                    raise ValueError(f"truncated counts: {tensor.name}")
                values = struct.unpack(f"<{EXPERTS}f", raw)
                if any(not math.isfinite(v) or v < 0 or not v.is_integer() for v in values):
                    raise ValueError(f"counts must be finite nonnegative integers: {tensor.name}")
                if any(v > expected_rows for v in values):
                    raise ValueError(f"expert counts exceed collected row bound: {tensor.name}")
                roles.append(tuple(int(v) for v in values))
            if roles[0] != roles[1] or roles[0] != roles[2]:
                raise ValueError(f"gate/up/down role mismatch at layer {layer}")
            if sum(roles[0]) != expected_visits:
                raise ValueError(f"unsupported/inconsistent calibration exposure: visit sum at layer {layer}: "
                                 f"{sum(roles[0])}, expected {expected_visits}; "
                                 "this profile requires complete chunk rows collected once per layer at top-k=8; "
                                 "chunk metadata is not a universal exact-token guarantee")
            freq.update({(layer, expert): count for expert, count in enumerate(roles[0])})
    sha256 = source_digest(path)
    if stat_signature(path) != before:
        raise ValueError("imatrix changed during read/hash")
    provenance = {
        "path": str(path.resolve()), "sha256": sha256, "bytes": before[2],
        "stat": dict(zip(("device", "inode", "size", "mtime_ns", "ctime_ns"), before)),
        "datasets": datasets, "chunk_count": chunks, "chunk_size": chunk_size,
        "collector_semantics": "Full chunk rows, one routing collection per trunk layer, top-k=8; "
                               "chunk_count*chunk_size is checked against every layer's count sum/8. "
                               "The collector derives chunk_count from max floor(counter/chunk_size), "
                               "not an explicit total-token counter; partial or imported collections can differ. "
                               "Metadata alone does not prove this for other collector configurations.",
        "schema_source": {"collector": "llama.cpp/tools/imatrix/imatrix.cpp:319-335,347-351,607-629",
                          "shared_route_ids": "llama.cpp/src/llama-graph.cpp:2212,2225,2326",
                          "counter_only_support": "counts suffice here; upstream writer also emits in_sum2"},
        "collected_rows_per_layer": expected_rows, "visits_per_layer": expected_visits,
        "total_trunk_visits": sum(freq.values()), "roles_counted": "gate only; up/down equality verified",
        "training_weight_fingerprint": "unknown: not recorded in this GGUF",
    }
    return freq, provenance


def read_mtp_jsonl(path):
    path = pathlib.Path(path)
    before = stat_signature(path)
    freq, selected, excluded = Counter(), 0, Counter()
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"invalid MTP JSONL row {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"MTP JSONL row {line_number} must be an object")
            kind, step, pos = row.get("kind"), row.get("step"), row.get("pos")
            if (kind not in ("prefill", "round", "draft") or type(step) is not int or step < 0
                    or type(pos) is not int or pos < 0 or "src" not in row):
                raise ValueError(f"invalid MTP JSONL kind/step/pos/src at row {line_number}")
            if "layer" in row and row["layer"] != 45:
                raise ValueError(f"MTP JSONL layer must be 45 at row {line_number}")
            ids, weights = row.get("ids"), row.get("w")
            if (not isinstance(ids, list) or len(ids) != 8
                    or any(type(i) is not int or not 0 <= i < EXPERTS for i in ids)
                    or len(set(ids)) != 8):
                raise ValueError(f"MTP JSONL ids must be 8 unique in-range integers at row {line_number}")
            if (not isinstance(weights, list) or len(weights) != 8
                    or any(type(w) not in (int, float) or w < 0 or w > sys.float_info.max
                           or not math.isfinite(w) for w in weights)):
                raise ValueError(f"MTP JSONL weights must be 8 finite nonnegative numbers at row {line_number}")
            if kind == "round" and step == 0:
                freq.update({(45, i): 1 for i in ids})
                selected += 1
            else:
                excluded[kind] += 1
    if not selected:
        raise ValueError("MTP JSONL has no round step0 catchup rows")
    sha256 = source_digest(path)
    if stat_signature(path) != before:
        raise ValueError("MTP trace changed during read/hash")
    return freq, {"status": "measured", "format": "route45-jsonl", "path": str(path.resolve()),
                  "sha256": sha256, "bytes": before[2], "draft_rows": 0,
                  "catchup_rows": selected, "visits": selected * 8,
                  "measurement_scope": "measuredteacherforcedcatchup-only",
                  "selection": "kind=round and step=0; layer45 implicit",
                  "excluded_rows": dict(excluded),
                  "warning": "Teacher-forced drafts and prefill excluded; not a measured acceptance-loop distribution",
                  "counts_source": "unique routed IDs only; weights not used"}


def read_mtp_trace(path):
    path = pathlib.Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_mtp_jsonl(path)
    before = stat_signature(path)
    freq, rows = Counter(), 0
    with path.open("rb") as fh:
        while raw := fh.read(72):
            if len(raw) != 72:
                raise ValueError("truncated MTP trace record")
            layer, k = struct.unpack_from("<ii", raw)
            ids = struct.unpack_from("<8i", raw, 8)
            weights = struct.unpack_from("<8f", raw, 40)
            if layer != 45 or k != 8:
                raise ValueError("MTP trace requires layer=45, k=8")
            if len(set(ids)) != 8 or any(not 0 <= i < EXPERTS for i in ids):
                raise ValueError("MTP trace ids must be unique and in range")
            if any(not math.isfinite(w) or w < 0 for w in weights):
                raise ValueError("MTP trace weights must be finite and nonnegative")
            freq.update({(45, i): 1 for i in ids})
            rows += 1
    if not rows:
        raise ValueError("MTP trace has no draft rows")
    sha256 = source_digest(path)
    if stat_signature(path) != before:
        raise ValueError("MTP trace changed during read/hash")
    return freq, {"status": "measured", "path": str(path.resolve()), "sha256": sha256,
                  "bytes": before[2], "draft_rows": rows, "visits": rows * 8,
                  "counts_source": "unique routed IDs only; weights not used"}


def coverage(ranked, freq):
    hits = sum(freq.get(pair, 0) for pair in ranked)
    total = sum(freq.values())
    hit = hits / total
    return {"trunk_hits": hits, "trunk_visits": total, "in_sample_trunk_hit": hit,
            "estimated_trunk_misses_per_row": (LAYERS - FIRST_MOE) * TOP_K * (1 - hit)}


def write_ranking(out, stem, ranked, n_layers):
    P.write_profile(out / (stem + ".strp"), ranked, n_layers, EXPERTS, FIRST_MOE)
    if P.read_profile(out / (stem + ".strp"), n_layers, EXPERTS, FIRST_MOE) != ranked:
        raise ValueError("STRP round trip mismatch")
    with (out / (stem + ".txt")).open("x", encoding="ascii") as fh:
        fh.writelines(f"{layer} {expert}\n" for layer, expert in ranked)


def build_profiles(imatrix, out, slots=(1600, 1650, 1700), include_mtp=False,
                   mtp_slots=32, mtp_trace=None):
    out = pathlib.Path(out)
    if out.exists():
        raise FileExistsError(f"refusing existing output directory: {out}")
    slots = tuple(slots)
    if not slots or any(type(s) is not int or not 1 <= s <= (LAYERS - FIRST_MOE) * EXPERTS for s in slots):
        raise ValueError("slots must be positive capacities within trunk pair count")
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate capacities")
    if type(mtp_slots) is not int or not 0 <= mtp_slots <= EXPERTS:
        raise ValueError("mtp-slots must be between 0 and 288")
    if mtp_trace and not include_mtp:
        raise ValueError("--mtp-trace requires --include-mtp")
    if include_mtp and any(s <= mtp_slots for s in slots):
        raise ValueError("capacity must be at least mtp reserve + 1")
    freq, provenance = read_imatrix(imatrix)
    trunk = P.rank_pairs(freq, n_layers=LAYERS, n_expert=EXPERTS, first_moe=FIRST_MOE)
    zero = P.rank_pairs({}, n_layers=LAYERS, n_expert=EXPERTS, first_moe=FIRST_MOE)
    variants = [("trunk-45", LAYERS, trunk, 0, {"status": "excluded", "reserved_slots": 0})]
    if include_mtp:
        if mtp_trace:
            mtp_freq, mtp = read_mtp_trace(mtp_trace)
        else:
            mtp_freq = Counter({(45, expert): freq[44, expert] for expert in range(EXPERTS)})
            mtp = {"status": "placeholder", "counts_source": "layer44proxy", "MTP": "unobserved",
                   "warning": "MTP proxy counts are not measured and never enter trunk coverage"}
        mtp_rank = P.rank_pairs(mtp_freq, n_layers=46, n_expert=EXPERTS, first_moe=45)
        ranked = mtp_rank[:mtp_slots] + trunk + mtp_rank[mtp_slots:]
        mtp["expert_frequencies"] = [mtp_freq.get((45, expert), 0) for expert in range(EXPERTS)]
        mtp["coverage_scope"] = "excluded from measured trunk hit numerator and denominator"
        mtp["reserved_slots"] = mtp_slots
        mtp["reserved_cache_bytes"] = mtp_slots * EXPERT_BYTES
        variants.append(("mtp-46-" + mtp["status"], 46, ranked, mtp_slots, mtp))
    report = {"schema": "glm-imatrix-profile-v1", "source": provenance,
              "prediction_scope": "in-sample trunk routing coverage only; not throughput or out-of-corpus hit",
              "ranking": "global frequency descending; make_profile.rank_pairs tie rule; equal 13MiB expert costs",
              "expert_bytes": EXPERT_BYTES,
              "frequencies": {str(layer): [freq[layer, expert] for expert in range(EXPERTS)]
                              for layer in range(FIRST_MOE, LAYERS)}, "profiles": []}
    out.mkdir(parents=True, exist_ok=False)
    for stem, n_layers, ranked, reserve, mtp in variants:
        write_ranking(out, stem + "-full", ranked, n_layers)
        profile = {"name": stem, "layers": n_layers, "full_pairs": len(ranked), "mtp": mtp,
                   "full_profile": stem + "-full.strp", "full_ranking_text": stem + "-full.txt", "capacities": []}
        for capacity in slots:
            prefix = ranked[:capacity]
            name = f"{stem}-{capacity}"
            write_ranking(out, name, prefix, n_layers)
            metrics = coverage(prefix, freq)
            baseline = zero[:capacity - reserve]
            metrics.update({"slots": capacity, "mtp_slots": reserve,
                            "cache_bytes": capacity * EXPERT_BYTES,
                            "per_layer_slots": {str(layer): sum(p[0] == layer for p in prefix)
                                                for layer in range(FIRST_MOE, n_layers)},
                            "zero_seed_same_trunk_budget": coverage(baseline, freq),
                            "zero_seed_same_total_capacity": coverage(zero[:capacity], freq),
                            "strp": name + ".strp", "ranking_text": name + ".txt"})
            profile["capacities"].append(metrics)
        report["profiles"].append(profile)
    with (out / "profiles.json").open("x", encoding="ascii") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True, help="fresh output directory; existing paths are refused")
    ap.add_argument("--slots", type=int, nargs="+", default=[1600, 1650, 1700])
    ap.add_argument("--include-mtp", action="store_true")
    ap.add_argument("--mtp-slots", type=int, default=32)
    ap.add_argument("--mtp-trace", help="native binary records or .jsonl route45 catchup trace")
    args = ap.parse_args()
    try:
        build_profiles(args.imatrix, args.out, args.slots, args.include_mtp, args.mtp_slots, args.mtp_trace)
    except (OSError, ValueError, struct.error) as exc:
        ap.exit(1, f"profile_imatrix: {exc}\n")
    print(f"wrote {args.out}: in-sample profiles; model weight fingerprint unknown")


if __name__ == "__main__":
    main()
