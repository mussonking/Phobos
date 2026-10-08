#!/usr/bin/env python3
"""Smoke test of glm_serve's line protocol, without the HTTP server.

usage: serve_smoke.py EXE PROMPT_TOKENS_TXT [engine args...]

1. GEN 24 on the prompt: T lines, DONE with reused 0.
2. Follow-up: prompt + the generated tokens + a user turn (the first request's prompt tail stands in for it):
   DONE must report reused = everything fed so far, and only the suffix read.
3. STOP mid-generation: DONE with finish "cancel".
4. The first prompt again (a retry): restored from its prompt-end snapshot (reused = all of it), same tokens.
5. A prompt diverging inside the first one: restored from the system-block snapshot (0 < reused < its length).
6. ERR on a malformed line, then QUIT.
DONE fields: gen prompt prompt_ms decode_ms finish acc off reused hits lookups.
"""
import subprocess
import sys
import time

USER, ASSISTANT = 154827, 154828   # <|user|>, <|assistant|>


def main():
    exe, toks_path, *args = sys.argv[1:]
    prompt = [int(x) for x in open(toks_path).read().split()]
    p = subprocess.Popen([exe, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    t0 = time.time()
    for line in p.stdout:
        print("<", line.rstrip())
        if line.startswith("READY"):
            break
    print(f"ready in {time.time() - t0:.0f} s")

    def gen(max_new, ids, stop_after=None):
        p.stdin.write(f"GEN {max_new} temperature=0 {','.join(map(str, ids))}\n")
        p.stdin.flush()
        out, first = [], None
        t = time.time()
        for line in p.stdout:
            f = line.split()
            if f[0] == "T":
                if first is None:
                    first = time.time() - t
                out.append(int(f[1]))
                if stop_after is not None and len(out) == stop_after:
                    p.stdin.write("STOP\n")
                    p.stdin.flush()
            elif f[0] in ("DONE", "ERR"):
                print("<", line.rstrip(), f"(TTFT {first if first is None else round(first, 2)} s)")
                return out, f
        raise SystemExit("engine died")

    out1, d1 = gen(24, prompt)
    assert d1[0] == "DONE" and int(d1[8]) == 0, d1
    follow = prompt + out1 + [USER] + prompt[-12:] + [ASSISTANT]
    out2, d2 = gen(16, follow)
    assert d2[0] == "DONE" and int(d2[8]) == len(prompt) + len(out1), d2
    out3, d3 = gen(200, follow + out2 + [USER, ASSISTANT], stop_after=5)
    assert d3[0] == "DONE" and d3[5] == "cancel" and int(d3[1]) <= 8, d3
    out4, d4 = gen(24, prompt)
    assert d4[0] == "DONE" and int(d4[8]) == len(prompt) and out4 == out1, (d4, out4, out1)
    _, d5 = gen(4, prompt[:-3])
    assert d5[0] == "DONE" and 0 < int(d5[8]) < len(prompt) - 3, d5
    p.stdin.write("GEN x\n")
    p.stdin.flush()
    print("<", p.stdout.readline().rstrip())
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    print("exit", p.wait(timeout=120))
    print("OK")


if __name__ == "__main__":
    main()
