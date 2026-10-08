#!/usr/bin/env python3
"""Needle-in-a-haystack through a running server's OpenAI endpoint (/v1/chat/completions), long-context check.

usage: needle_http.py NEEDLE_DIR [--url http://127.0.0.1:8142] [--key-file FILE] [--ids 100k-d50,120k-d90]
                      [--max-tokens N]

NEEDLE_DIR is make_needles.py's output: each prompt's user message is taken back from its rendered prompt.txt (the
text between <|user|> and <|assistant|>) and sent as one chat message; the server renders the template itself.
Per prompt: the server's prompt token count, total time, and whether the code word is in the answer (reasoning
included). Exit status 1 if a word is missed.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--url", default="http://127.0.0.1:8142")
    ap.add_argument("--key-file")
    ap.add_argument("--ids", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    headers = {"content-type": "application/json"}
    if a.key_file:
        headers["authorization"] = "Bearer " + open(a.key_file).read().strip()
    rows = json.load(open(os.path.join(a.dir, "needles.json")))
    if a.ids:
        want = a.ids.split(",")
        rows = [x for x in rows if x["id"] in want]
    found = 0
    for x in rows:
        text = open(os.path.join(a.dir, x["id"], "prompt.txt"), encoding="utf-8").read()
        user = text[text.index("<|user|>") + len("<|user|>"):text.rindex("<|assistant|>")]
        body = {"model": "default", "max_tokens": a.max_tokens, "stream": False,
                "messages": [{"role": "user", "content": user}]}
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=7200) as r:
            res = json.load(r)
        dt = time.time() - t0
        msg = res["choices"][0]["message"]
        answer = (msg.get("content") or "").strip()
        ok = x["word"] in answer + (msg.get("reasoning_content") or "")
        found += ok
        u = res.get("usage") or {}
        print(f"{x['id']:>10} prompt {u.get('prompt_tokens')} tokens (make_needles: {x['n_tokens']}), "
              f"{u.get('completion_tokens')} generated, {dt:.0f} s, finish {res['choices'][0].get('finish_reason')}: "
              f"{'FOUND' if ok else 'missed'} {x['word']}  answer {answer[:60]!r}", flush=True)
    print(f"{found}/{len(rows)} found")
    return 0 if found == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
