#!/usr/bin/env python3
"""Run a verified GLM quant recipe, refusing overwrite and verifying output."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from quant_recipe import load_recipe, verify_dry, verify_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=pathlib.Path, required=True)
    p.add_argument("--dry-log", type=pathlib.Path, required=True)
    args = p.parse_args()
    recipe = load_recipe(args.recipe)
    dry_messages = io.StringIO()
    with contextlib.redirect_stdout(dry_messages):
        verify_dry(argparse.Namespace(recipe=args.recipe, log=args.dry_log))
    output = pathlib.Path(recipe["output"])
    if output.exists():
        sys.exit(f"Refusing to overwrite {output}; inspect the previous run first")
    required = recipe["expected_payload_bytes"] + 16 * 1024**3
    if shutil.disk_usage(output.parent).free < required:
        sys.exit(f"Insufficient output disk space: need {required} bytes with safety margin")
    state_path = args.recipe.parent / "quant-state.json"
    log_path = args.recipe.parent / "quantize.log"
    if log_path.exists() or state_path.exists():
        sys.exit("quantize.log/state already exists; inspect it, do not overwrite")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    started = time.time()
    state = {"status": "running", "started": started, "output": str(output),
             "supervisor_pid": os.getpid(), "log": str(log_path)}
    with log_path.open("x") as log:
        log.write(dry_messages.getvalue())
        log.flush()
        proc = subprocess.Popen(["nice", "-n", "10"] + recipe["argv"],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        state["pid"] = proc.pid
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        rc = proc.wait()
    state["exit_code"] = rc
    state["elapsed_s"] = time.time() - started
    state["status"] = "quantizer_failed" if rc else "verifying"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    if rc:
        sys.exit(f"Quantizer failed ({rc}); inspect {log_path}")
    verification_messages = io.StringIO()
    try:
        with contextlib.redirect_stdout(verification_messages):
            verify_output(argparse.Namespace(recipe=args.recipe, output=output))
    except Exception as exc:
        state["status"] = "verification_failed"
        state["error"] = str(exc)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        raise
    state["status"] = "verified"
    state["finished"] = time.time()
    state["file_bytes"] = output.stat().st_size
    state["type_counts"] = recipe["type_counts"]
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    message = "QUANT_DONE verified " + json.dumps(state) + "\n"
    with log_path.open("a") as log:
        log.write(verification_messages.getvalue())
        log.write(message)
    try:
        sys.stdout.write(message)
        sys.stdout.flush()
    except BrokenPipeError:
        # Keep a closed SSH console separate from the durable artifact verdict,
        # and prevent a second broken-pipe error during interpreter shutdown.
        sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    main()
