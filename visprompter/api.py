"""Embedded FastAPI app: serves the bundled VisPrompter UI + reads a store.
Optionally runs in-process capture (when a model is loaded via `visprompter serve`)."""
from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "ui" / "static"


def create_app(store, capture=None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="VisPrompter")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/api/infer")
    def infer(body: dict):
        if capture is None:
            raise HTTPException(503, "this server is view-only (no model loaded). "
                                     "Run `visprompter serve --model <id> --port <N>` to enable inference.")
        s = capture.capture(body.get("prompt", ""), int(body.get("max_new_tokens", 32)))
        return {"session_id": s["session_id"], "generated_text": s["gen_text"],
                "n_layers": s["n_layers"], "n_prompt_tok": s["n_prompt"], "n_gen_tok": s["n_gen"]}

    @app.get("/api/sessions")
    def sessions():
        return store.list_sessions()

    @app.get("/api/sessions/{sid}")
    def session(sid: str):
        m = store.get_session(sid)
        if not m:
            raise HTTPException(404, "session not found")
        return m

    @app.get("/api/sessions/{sid}/surface")
    def surface(sid: str, metric: str = Query("residual_delta")):
        if metric not in ("residual_delta", "hidden_norm", "attn_entropy"):
            raise HTTPException(400, "bad metric")
        return store.get_surface(sid, metric)

    @app.get("/api/sessions/{sid}/logitlens")
    def logitlens(sid: str, token_pos: int):
        return store.get_logitlens(sid, token_pos)

    @app.get("/health")
    def health():
        return {"status": "ok", "inference": capture is not None}

    # serve the bundled SPA (if built); SPA fallback to index.html
    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "no such endpoint")
            f = STATIC_DIR / path
            return FileResponse(f if f.is_file() else STATIC_DIR / "index.html")

    return app
