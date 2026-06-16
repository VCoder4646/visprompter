"""visprompter CLI:  run | ui | serve

  visprompter run   --model google/gemma-4-E4B-it --prompt "Why is the sky blue?" --store ./vp.db
  visprompter ui    --store ./vp.db --port 9000          # view-only visualizer on your port
  visprompter serve --model <id> --store ./vp.db --port 9000   # viz + live inference
"""
from __future__ import annotations

import argparse
import os

from .store import open_store


def _add_store(p):
    p.add_argument("--store", default=os.environ.get("VISPROMPTER_STORE", "./visprompter.db"),
                   help="SQLite path (default ./visprompter.db) or postgresql:// DSN")


def cmd_run(args):
    from .capture import OmniVisCapture
    store = open_store(args.store)
    cap = OmniVisCapture.from_pretrained(args.model, store=store, hf_token=os.environ.get("HF_TOKEN"))
    for prompt in args.prompt:
        s = cap.capture(prompt, max_new_tokens=args.max_new_tokens)
        print(f"[{s['session_id']}] {prompt!r} -> {s['gen_text'][:80]!r}")
    print(f"stored in {args.store}. View it with:  visprompter ui --store {args.store} --port {args.port}")


def cmd_ui(args):
    import uvicorn
    from .api import create_app
    app = create_app(open_store(args.store), capture=None)
    print(f"VisPrompter (view-only) at http://0.0.0.0:{args.port}  (store: {args.store})")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_serve(args):
    import uvicorn
    from .api import create_app
    from .capture import OmniVisCapture
    store = open_store(args.store)
    cap = OmniVisCapture.from_pretrained(args.model, store=store, hf_token=os.environ.get("HF_TOKEN"))
    app = create_app(store, capture=cap)
    print(f"VisPrompter (live inference) at http://0.0.0.0:{args.port}  model: {args.model}")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    ap = argparse.ArgumentParser(prog="visprompter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="capture one or more prompts into the store")
    r.add_argument("--model", required=True)
    r.add_argument("--prompt", required=True, nargs="+")
    r.add_argument("--max-new-tokens", type=int, default=32)
    r.add_argument("--port", type=int, default=9000)  # only for the hint message
    _add_store(r); r.set_defaults(fn=cmd_run)

    u = sub.add_parser("ui", help="serve the visualizer (view-only) on a port")
    u.add_argument("--port", type=int, default=9000)
    u.add_argument("--host", default="0.0.0.0")
    _add_store(u); u.set_defaults(fn=cmd_ui)

    s = sub.add_parser("serve", help="serve the visualizer + live inference on a port")
    s.add_argument("--model", required=True)
    s.add_argument("--port", type=int, default=9000)
    s.add_argument("--host", default="0.0.0.0")
    _add_store(s); s.set_defaults(fn=cmd_serve)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
