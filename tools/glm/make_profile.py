"""Write GLM ranked expert cache profiles in Strata's STRP v1 wire format.

The header is STRP + (version, layers, experts, slots, ranked_count) as little
endian u32, followed by u16 (layer,expert) pairs and an int32 slot table. The
trailing table is a slot map, not frequencies. Zero-frequency experts remain
valid ranked pairs. Dense layers never appear in rankings and have -1 slots.
Routing traces contain int32 layer, int32 k, k int32 ids, then k float32 weights.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import struct
from collections import Counter

MAGIC, VERSION = b"STRP", 1
N_LAYER, N_EXPERT, FIRST_MOE = 46, 288, 3


def validate_geometry(n_layers, n_expert, first_moe):
    if not (0 <= first_moe < n_layers <= 65536 and 0 < n_expert <= 65536):
        raise ValueError("geometry must fit uint16 pair ids and contain routed layers")


def validate_pairs(ranked, n_layers, n_expert, first_moe):
    validate_geometry(n_layers, n_expert, first_moe)
    seen = set()
    for pair in ranked:
        layer, expert = pair
        if not (first_moe <= layer < n_layers and 0 <= expert < n_expert):
            raise ValueError(f"invalid routed pair {pair}")
        if pair in seen:
            raise ValueError(f"duplicate ranked pair {pair}")
        seen.add(pair)


def read_profile(path, n_layers=N_LAYER, n_expert=N_EXPERT, first_moe=FIRST_MOE):
    blob = pathlib.Path(path).read_bytes()
    if len(blob) < 24 or blob[:4] != MAGIC:
        raise ValueError(f"{path}: not a STRP profile")
    version, nl, ne, slots, count = struct.unpack_from("<5I", blob, 4)
    if version != VERSION or (nl, ne) != (n_layers, n_expert):
        raise ValueError(f"{path}: incompatible profile version or geometry")
    if slots != count or len(blob) != 24 + 4 * count + 4 * nl * ne:
        raise ValueError(f"{path}: invalid slots, count, or file length")
    ranked = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(count)]
    validate_pairs(ranked, nl, ne, first_moe)
    expected = [-1] * (nl * ne)
    for slot, (layer, expert) in enumerate(ranked):
        expected[layer * ne + expert] = slot
    actual = struct.unpack_from(f"<{nl * ne}i", blob, 24 + 4 * count)
    if tuple(expected) != actual:
        raise ValueError(f"{path}: inconsistent slot table")
    return ranked


def read_trace(path, n_layers=N_LAYER, n_expert=N_EXPERT, first_moe=FIRST_MOE):
    validate_geometry(n_layers, n_expert, first_moe)
    blob = pathlib.Path(path).read_bytes()
    off, freq = 0, Counter()
    while off < len(blob):
        if len(blob) - off < 8:
            raise ValueError(f"{path}: truncated trace header at {off}")
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        if not (first_moe <= layer < n_layers and 0 < k <= n_expert):
            raise ValueError(f"{path}: invalid trace layer/k {layer}/{k}")
        if len(blob) - off < 8 * k:
            raise ValueError(f"{path}: truncated trace record at {off - 8}")
        ids = struct.unpack_from(f"<{k}i", blob, off)
        weights = struct.unpack_from(f"<{k}f", blob, off + 4 * k)
        pairs = [(layer, expert) for expert in ids]
        validate_pairs(pairs, n_layers, n_expert, first_moe)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError(f"{path}: nonfinite or negative routing weight")
        freq.update(pairs)
        off += 8 * k
    return freq


def rank_pairs(freq, base=(), n_layers=N_LAYER, n_expert=N_EXPERT, first_moe=FIRST_MOE):
    validate_pairs(base, n_layers, n_expert, first_moe)
    validate_pairs(list(freq), n_layers, n_expert, first_moe)
    if any(not isinstance(c, int) or c < 0 for c in freq.values()):
        raise ValueError("routing frequencies must be nonnegative integers")
    ranked, seen = [], set()

    def take(pairs):
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                ranked.append(pair)

    take(base)
    take(p for p, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0])))
    take((layer, expert) for expert in range(n_expert) for layer in range(first_moe, n_layers))
    return ranked


def write_profile(path, ranked, n_layers=N_LAYER, n_expert=N_EXPERT, first_moe=FIRST_MOE):
    validate_pairs(ranked, n_layers, n_expert, first_moe)
    table = [-1] * (n_layers * n_expert)
    for slot, (layer, expert) in enumerate(ranked):
        table[layer * n_expert + expert] = slot
    with pathlib.Path(path).open("xb") as fh:
        fh.write(MAGIC + struct.pack("<5I", VERSION, n_layers, n_expert, len(ranked), len(ranked)))
        for pair in ranked:
            fh.write(struct.pack("<HH", *pair))
        fh.write(struct.pack(f"<{len(table)}i", *table))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--base", help="optional same-geometry STRP ranking to preserve first")
    ap.add_argument("--out", required=True, help="new profile path (existing files are refused)")
    ap.add_argument("--n-layers", type=int, default=N_LAYER)
    ap.add_argument("--n-expert", type=int, default=N_EXPERT)
    ap.add_argument("--first-moe-layer", type=int, default=FIRST_MOE)
    args = ap.parse_args()
    geometry = (args.n_layers, args.n_expert, args.first_moe_layer)
    try:
        validate_geometry(*geometry)
        base = read_profile(args.base, *geometry) if args.base else []
        freq = Counter()
        for trace in args.traces:
            freq.update(read_trace(trace, *geometry))
        ranked = rank_pairs(freq, base, *geometry)
        write_profile(args.out, ranked, *geometry)
        if read_profile(args.out, *geometry) != ranked:
            raise ValueError("profile round trip mismatch")
    except (OSError, ValueError) as exc:
        ap.exit(1, f"make_profile: {exc}\n")
    print(f"wrote {args.out}: {len(ranked)} routed pairs, geometry {args.n_layers}x{args.n_expert}, "
          f"{sum(freq.values())} observed routes (zero-frequency pairs included)")


if __name__ == "__main__":
    main()
