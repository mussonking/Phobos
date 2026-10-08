#!/usr/bin/env python3
"""Mini GLM-5.3-Flash test model: a few blocks of the real GGUF, byte for byte, renumbered.

usage: mini_gguf.py SRC.gguf OUT.gguf [--blocks 0,1,4,3] [--dense-lead 2] [--mtp 45]
                    [--profile SRC.strp --profile-out OUT.strp]

Same formats as the source (IQ4_XS gate/up, IQ4_NL down, Q8_0 dense), same tokenizer and metadata, the real expert
count per routed block: the engine loads it through the same code (load_model reads the depth from block_count,
nextn_predict_layers and leading_dense_block_count). --blocks lists the source trunk blocks in their new order; the
layer kinds must still follow the engine's schedule (MLA + DSA where index % 4 == 3, dense below --dense-lead), so the
default is 0, 1 dense KDA, 4 routed KDA, 3 routed MLA + DSA, then the MTP block (source 45) as block 4. The output is
garbage text by design: it is a numerics and code-path test model (kernel parities, F64 vs GPU, serve), not a model.
Per-block metadata arrays (head_count_kv, swiglu clamps) are remapped the same way; provenance goes to
strata.mini.* keys. --profile remaps a STRP expert ranking ((layer, expert) pairs) to the new block numbers.
"""
import argparse
import hashlib
import os
import struct
import sys

import gguf

ARCH = "glm5-next"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--blocks", default="0,1,4,3")
    ap.add_argument("--dense-lead", type=int, default=2)
    ap.add_argument("--mtp", type=int, default=45)
    ap.add_argument("--profile")
    ap.add_argument("--profile-out")
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    r = gguf.GGUFReader(a.src, "r")

    def f(k):
        return r.fields[k].contents()

    if f("general.architecture") != ARCH:
        sys.exit(f"{a.src}: not a {ARCH} GGUF")
    n_src, n_mtp, lead_src = f(f"{ARCH}.block_count"), f(f"{ARCH}.nextn_predict_layers"), f(f"{ARCH}.leading_dense_block_count")
    if n_mtp != 1 or a.mtp != n_src - 1:
        sys.exit(f"expected one MTP block at {n_src - 1}")
    kv = f(f"{ARCH}.attention.head_count_kv")
    n = len(blocks)
    src_of = blocks + [a.mtp]          # new block i <- source block src_of[i]
    for i, s in enumerate(blocks):     # the engine's schedule, checked here so a bad --blocks fails before writing
        if not 0 <= s < a.mtp:
            sys.exit(f"--blocks: {s} is not a trunk block")
        if (s < lead_src) != (i < a.dense_lead):
            sys.exit(f"--blocks: source {s} is {'dense' if s < lead_src else 'routed'}, position {i} must be "
                     f"{'dense' if i < a.dense_lead else 'routed'} (--dense-lead {a.dense_lead})")
        if (kv[s] != 0) != (i % 4 == 3):
            sys.exit(f"--blocks: source {s} is {'MLA' if kv[s] else 'KDA'}, position {i} must be "
                     f"{'MLA' if i % 4 == 3 else 'KDA'}")

    w = gguf.GGUFWriter(a.out, ARCH)
    per_block = {}
    for field in r.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        t = field.types[0]
        sub = field.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        v = field.contents()
        if field.name == f"{ARCH}.block_count":
            v = n + 1
        elif field.name == f"{ARCH}.leading_dense_block_count":
            v = a.dense_lead
        elif field.name == "general.name":
            v = f"{v} mini {n}+1 blocks (test model)"
        elif t == gguf.GGUFValueType.ARRAY and field.name.startswith(ARCH + ".") and len(v) == n_src:
            v = [v[s] for s in src_of]
            per_block[field.name] = v
        w.add_key_value(field.name, v, t, sub_type=sub)
    w.add_key_value("strata.mini.source_blocks", src_of, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
    w.add_key_value("strata.mini.source_file", os.path.basename(a.src), gguf.GGUFValueType.STRING)
    w.add_key_value("strata.mini.source_size", os.path.getsize(a.src), gguf.GGUFValueType.UINT64)

    new_of = {s: i for i, s in enumerate(src_of)}
    keep = []
    for t in r.tensors:
        p = t.name.split(".")
        if p[0] == "blk":
            s = int(p[1])
            if s not in new_of:
                continue
            name = ".".join(["blk", str(new_of[s])] + p[2:])
        else:
            name = t.name
        keep.append((name, t))
    keep.sort(key=lambda x: (x[0].split(".")[0] != "blk", int(x[0].split(".")[1]) if x[0].startswith("blk.") else 0))
    total = 0
    for name, t in keep:
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += int(t.n_bytes)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    h = hashlib.sha256()
    for i, (name, t) in enumerate(keep):
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        h.update(name.encode())
        h.update(memoryview(t.data).cast("B"))
        if i % 50 == 0:
            print(f"  {i}/{len(keep)} tensors", flush=True)
    w.close()
    print(f"{a.out}: {len(keep)} tensors, {total / 1e9:.2f} GB of tensor data, blocks {src_of} -> 0..{n}, "
          f"dense lead {a.dense_lead}; per-block arrays remapped: {', '.join(sorted(per_block))}")
    print(f"tensor content sha256 (names + bytes, in file order): {h.hexdigest()}")

    if a.profile:
        b = open(a.profile, "rb").read()
        if b[:4] != b"STRP":
            sys.exit(f"{a.profile}: not a STRP profile")
        ver, _, ne, _, count = struct.unpack_from("<5I", b, 4)
        pairs = [struct.unpack_from("<2H", b, 24 + 4 * i) for i in range(count)]
        out = [(new_of[il], e) for il, e in pairs if il in new_of and il >= lead_src]
        n_routed = (n + 1 - a.dense_lead) * ne
        with open(a.profile_out, "wb") as o:
            o.write(b"STRP" + struct.pack("<5I", ver, n + 1, ne, n_routed, len(out)))
            for il, e in out:
                o.write(struct.pack("<2H", il, e))
        print(f"{a.profile_out}: {len(out)} of {count} ranked experts kept (expected {n_routed})")


if __name__ == "__main__":
    main()
