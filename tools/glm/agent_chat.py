#!/usr/bin/env python3
"""Claude-Code-style agent session against a Strata server's Anthropic endpoint (/v1/messages, streamed).

usage: agent_chat.py [--url http://127.0.0.1:8142] [--max-tokens N] [--turns N] [--budget N] [--out FILE]
                     [--system-file FILE]   (appended to the system prompt as project documentation, to reach a
                                             Claude-Code-sized system prompt)
                     [--key-file FILE]      (the server's API key, sent as x-api-key)

A coding-agent system prompt, four tools (Bash, Read, Grep, Edit) over a small in-memory repository, and a task
that needs several tool calls. Each request resends the whole conversation (thinking blocks included, as Claude
Code does); tool results come from the fake repository. Per request: time to the first streamed event of the
answer (TTFT, client side), output tokens and their rate after the first one, the stop reason, the tool calls.
The engine's own split (prompt tokens reused / read, ms) is in its log (glm_serve lines).
"""
import argparse
import json
import re
import sys
import time
import urllib.request

SYSTEM = """You are an interactive CLI coding agent. You help the user with software engineering tasks in their \
repository using the tools below. Be concise. Use tools to look at the code before answering; never guess file \
contents. When the task is done, answer in plain text without calling a tool.

Working directory: /work/app (a small Python web service). Platform: linux."""

TOOLS = [
    {"name": "Bash", "description": "Run a shell command in the working directory and return its output.",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string", "description": "The command to run"}}, "required": ["command"]}},
    {"name": "Read", "description": "Read a file and return its content with line numbers.",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string", "description": "Absolute path of the file"}}, "required": ["file_path"]}},
    {"name": "Grep", "description": "Search file contents with a regular expression; returns path:line:text.",
     "input_schema": {"type": "object", "properties": {
         "pattern": {"type": "string", "description": "The regular expression"},
         "path": {"type": "string", "description": "Directory or file to search (default: the working directory)"}},
         "required": ["pattern"]}},
    {"name": "Edit", "description": "Replace an exact string in a file.",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
         "required": ["file_path", "old_string", "new_string"]}},
]

REPO = {
    "/work/app/README.md": "# app\nA small HTTP service. Run: python -m app.server\n",
    "/work/app/app/__init__.py": "",
    "/work/app/app/config.py": (
        "import os\n\nDEFAULTS = {\n    \"host\": \"0.0.0.0\",\n    \"port\": 8080,\n    \"workers\": 4,\n}\n\n\n"
        "def load_config(env=os.environ):\n    cfg = dict(DEFAULTS)\n    if \"APP_PORT\" in env:\n"
        "        cfg[\"port\"] = int(env[\"APP_PORT\"])\n    return cfg\n"),
    "/work/app/app/server.py": (
        "from http.server import HTTPServer, BaseHTTPRequestHandler\n\nfrom app.config import load_config\n\n\n"
        "class Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n"
        "        self.end_headers()\n        self.wfile.write(b\"ok\")\n\n\n"
        "def main():\n    cfg = load_config()\n    HTTPServer((cfg[\"host\"], cfg[\"port\"]), Handler).serve_forever()\n"),
    "/work/app/tests/test_config.py": (
        "from app.config import load_config\n\n\ndef test_default_port():\n    assert load_config({})[\"port\"] == 8080\n"),
}

TASK = ("What port does the service listen on by default, and how can it be changed? Then change the default "
        "port to 9090 and update the test accordingly.")


def run_tool(name, args):
    if name == "Read":
        text = REPO.get(args.get("file_path", ""))
        if text is None:
            return "Error: file not found"
        return "".join(f"{i + 1:>5}\t{line}\n" for i, line in enumerate(text.splitlines()))
    if name == "Grep":
        try:
            rx = re.compile(args.get("pattern", ""))
        except re.error as e:
            return f"Error: {e}"
        base = args.get("path") or "/work/app"
        hits = [f"{p}:{i + 1}:{line}" for p, t in REPO.items() if p.startswith(base)
                for i, line in enumerate(t.splitlines()) if rx.search(line)]
        return "\n".join(hits) or "No matches"
    if name == "Edit":
        p = args.get("file_path", "")
        if p not in REPO or args.get("old_string", "") not in REPO[p]:
            return "Error: old_string not found in file"
        REPO[p] = REPO[p].replace(args["old_string"], args.get("new_string", ""), 1)
        return f"The file {p} has been updated."
    if name == "Bash":
        cmd = args.get("command", "")
        if cmd.startswith(("ls", "find", "tree")):
            return "\n".join(sorted(REPO))
        if cmd.startswith("cat "):
            return REPO.get(cmd.split()[1] if cmd.split()[1].startswith("/") else "/work/app/" + cmd.split()[1],
                            "cat: no such file")
        if "pytest" in cmd:
            ok = ("9090" in REPO["/work/app/tests/test_config.py"]) == ("9090" in REPO["/work/app/app/config.py"])
            return "1 passed in 0.01s" if ok else "1 failed in 0.01s"
        return "(no output)"
    return f"Error: unknown tool {name}"


def stream(url, body, key=None):
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    first = None
    blocks, cur, usage, stop = [], None, {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            kind = ev.get("type")
            if kind == "content_block_start":
                cur = dict(ev["content_block"])
                if cur["type"] == "tool_use":
                    cur["_json"] = ""
            elif kind == "content_block_delta":
                if first is None:
                    first = time.time()
                d = ev["delta"]
                if d["type"] == "text_delta":
                    cur["text"] = cur.get("text", "") + d["text"]
                elif d["type"] == "thinking_delta":
                    cur["thinking"] = cur.get("thinking", "") + d["thinking"]
                elif d["type"] == "input_json_delta":
                    cur["_json"] += d["partial_json"]
            elif kind == "content_block_stop":
                if cur.get("type") == "tool_use":
                    cur["input"] = json.loads(cur.pop("_json") or "{}")
                blocks.append(cur)
                cur = None
            elif kind == "message_delta":
                usage.update(ev.get("usage") or {})
                stop = ev.get("delta", {}).get("stop_reason")
            elif kind == "error":
                raise SystemExit(f"server error: {ev}")
    t1 = time.time()
    return blocks, usage, stop, t0, first, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8142")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--budget", type=int, default=1024, help="thinking budget_tokens (< 2048 = low effort)")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--out", help="write the per-request table as JSON")
    ap.add_argument("--system-file", help="text appended to the system prompt (project documentation)")
    ap.add_argument("--key-file", help="file holding the server's API key (sent as x-api-key)")
    a = ap.parse_args()
    key = open(a.key_file).read().strip() if a.key_file else None
    messages = [{"role": "user", "content": TASK}]
    system = SYSTEM
    if a.system_file:
        with open(a.system_file, encoding="utf-8") as f:
            system += "\n\n# Project documentation\n\n" + f.read()
    rows = []
    for turn in range(a.turns):
        body = {"model": "default", "max_tokens": a.max_tokens, "system": system, "tools": TOOLS, "stream": True,
                "messages": messages, "thinking": {"type": "enabled", "budget_tokens": a.budget}}
        blocks, usage, stop, t0, first, t1 = stream(a.url, body, key)
        n_out = int(usage.get("output_tokens") or 0)
        ttft = (first - t0) if first else None
        rate = (n_out - 1) / (t1 - first) if first and n_out > 1 and t1 > first else None
        calls = [b for b in blocks if b["type"] == "tool_use"]
        text = "".join(b.get("text", "") for b in blocks if b["type"] == "text")
        row = {"turn": turn, "ttft_s": ttft, "out_tokens": n_out, "decode_tok_s": rate, "stop": stop,
               "total_s": t1 - t0, "tools": [(c["name"], c["input"]) for c in calls]}
        rows.append(row)
        print(f"turn {turn}: TTFT {ttft if ttft is None else round(ttft, 2)} s, {n_out} tokens, "
              f"{rate if rate is None else round(rate, 1)} tok/s, stop {stop}, tools {row['tools']}", flush=True)
        if text.strip():
            print("  text:", text.strip()[:400].replace("\n", " | "), flush=True)
        messages.append({"role": "assistant", "content": blocks})
        if not calls:
            break
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c["id"], "content": run_tool(c["name"], c["input"])} for c in calls]})
    print("final config.py:", json.dumps(REPO["/work/app/app/config.py"]))
    print("final test:", json.dumps(REPO["/work/app/tests/test_config.py"]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
    def median(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else float("nan")

    print(f"first TTFT {rows[0]['ttft_s']:.2f} s; follow-up TTFT median {median(r['ttft_s'] for r in rows[1:]):.2f} s "
          f"over {len(rows) - 1} requests; decode tok/s median {median(r['decode_tok_s'] for r in rows):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
