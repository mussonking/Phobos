#!/usr/bin/env python3
"""Detached final verification fallback; never restart, stop or rewrite a quant.

Wait for the existing quantizer process, then independently validate the GGUF
into quant-final.json. The original supervisor retains its own state even if
its stdout breaks after an SSH timeout. Final verification records an unknown
quantizer exit code rather than inventing a successful exit status.
"""
import argparse
import json
import os
import pathlib
import sys
import time

from quant_recipe import load_recipe, verify_output


def alive(pid):
    try:
        os.kill(pid, 0)
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2:].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=pathlib.Path, required=True)
    args = p.parse_args()
    recipe = load_recipe(args.recipe)
    path = args.recipe.parent / "quant-state.json"
    initial = json.loads(path.read_text())
    final_path = args.recipe.parent / "quant-final.json"
    if final_path.exists():
        raise FileExistsError("detached final verification already exists; inspect it")
    end = time.time() + 8 * 3600
    while time.time() < end:
        if alive(initial["pid"]):
            time.sleep(30)
            continue
        # The original quantizer no longer writes the payload. A live or broken
        # SSH supervisor must not invalidate the independent artifact check.
        output = pathlib.Path(recipe["output"])
        state = {"output": str(output), "quantizer_pid": initial["pid"],
                 "exit_code": None, "verification_pid": os.getpid(),
                 "elapsed_s": time.time() - initial["started"]}
        try:
            verify_output(argparse.Namespace(recipe=args.recipe, output=output))
            state.update(status="verified", finished=time.time(),
                         file_bytes=output.stat().st_size, type_counts=recipe["type_counts"])
        except Exception as exc:
            state.update(status="verification_failed", error=str(exc))
        # Separate file: never race the original supervisor's state writer.
        with final_path.open("x") as f:
            f.write(json.dumps(state, indent=2) + "\n")
        sys.stdout.write("DETACHED_FINAL " + json.dumps(state) + "\n")
        return 0 if state["status"] == "verified" else 1
    sys.stderr.write("Fallback deadline reached; quant/state left untouched\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
