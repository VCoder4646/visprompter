"""Pluggable storage for OmniVis sessions.

Default: SQLite (zero-setup, fully standalone). Optional: Postgres (share with a
bigger stack). Both expose the same Python-level interface so the API/UI don't
care which backend is used. Arrays are JSON in SQLite, native arrays in Postgres.
"""
from __future__ import annotations

import json
from typing import List, Optional

METRICS = ("residual_delta", "hidden_norm", "attn_entropy")


def open_store(uri: str):
    """uri: a postgresql://... DSN -> PostgresStore, else a file path -> SQLiteStore."""
    if uri.startswith("postgres://") or uri.startswith("postgresql://"):
        return PostgresStore(uri)
    return SQLiteStore(uri)


# ----------------------------- SQLite -----------------------------
class SQLiteStore:
    def __init__(self, path: str):
        import sqlite3
        self.path = path
        self._sqlite3 = sqlite3
        self.init_schema()

    def _conn(self):
        c = self._sqlite3.connect(self.path)
        c.row_factory = self._sqlite3.Row
        return c

    def init_schema(self):
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS infer_sessions(
              session_id TEXT PRIMARY KEY, prompt TEXT, model_id TEXT, n_layers INT,
              n_prompt_tok INT, n_gen_tok INT, generated_text TEXT,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS infer_tokens(
              session_id TEXT, token_pos INT, token_str TEXT, token_id INT, is_prompt INT,
              PRIMARY KEY(session_id, token_pos));
            CREATE TABLE IF NOT EXISTS infer_layer_steps(
              session_id TEXT, token_pos INT, layer_idx INT,
              residual_delta REAL, hidden_norm REAL, attn_entropy REAL,
              PRIMARY KEY(session_id, token_pos, layer_idx));
            CREATE TABLE IF NOT EXISTS infer_logitlens(
              session_id TEXT, token_pos INT, layer_idx INT, top_token TEXT, top_prob REAL,
              topk_tokens TEXT, topk_probs TEXT,
              PRIMARY KEY(session_id, token_pos, layer_idx));
            """)

    def write_session(self, s):
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO infer_sessions
                (session_id,prompt,model_id,n_layers,n_prompt_tok,n_gen_tok,generated_text)
                VALUES (?,?,?,?,?,?,?)""",
                (s["session_id"], s["prompt"], s["model_id"], s["n_layers"],
                 s["n_prompt"], s["n_gen"], s["gen_text"]))
            c.executemany("INSERT OR REPLACE INTO infer_tokens VALUES (?,?,?,?,?)",
                          [(s["session_id"], p, ts, ti, int(b)) for (p, ts, ti, b) in s["tokens"]])
            c.executemany("INSERT OR REPLACE INTO infer_layer_steps VALUES (?,?,?,?,?,?)",
                          [(s["session_id"], p, l, d, n, e) for (p, l, d, n, e) in s["layer_rows"]])
            c.executemany("INSERT OR REPLACE INTO infer_logitlens VALUES (?,?,?,?,?,?,?)",
                          [(s["session_id"], p, l, tt, tp, json.dumps(tks), json.dumps(tps))
                           for (p, l, tt, tp, tks, tps) in s["lens_rows"]])

    def list_sessions(self):
        with self._conn() as c:
            rows = c.execute("""SELECT session_id,prompt,model_id,n_layers,n_prompt_tok,
                n_gen_tok,generated_text,created_at FROM infer_sessions
                ORDER BY created_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def get_session(self, sid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM infer_sessions WHERE session_id=?", (sid,)).fetchone()
            if not r:
                return None
            meta = dict(r)
            toks = c.execute("SELECT token_pos,token_str,token_id,is_prompt FROM infer_tokens "
                             "WHERE session_id=? ORDER BY token_pos", (sid,)).fetchall()
            meta["tokens"] = [{"token_pos": t["token_pos"], "token_str": t["token_str"],
                               "token_id": t["token_id"], "is_prompt": bool(t["is_prompt"])} for t in toks]
        return meta

    def get_surface(self, sid, metric):
        with self._conn() as c:
            rows = c.execute(f"SELECT layer_idx,token_pos,{metric} v FROM infer_layer_steps "
                             "WHERE session_id=? ORDER BY layer_idx,token_pos", (sid,)).fetchall()
        return _surface_payload(metric, rows)

    def get_logitlens(self, sid, token_pos):
        with self._conn() as c:
            rows = c.execute("""SELECT layer_idx,top_token,top_prob,topk_tokens,topk_probs
                FROM infer_logitlens WHERE session_id=? AND token_pos=? ORDER BY layer_idx""",
                (sid, token_pos)).fetchall()
        return [{"layer_idx": r["layer_idx"], "top_token": r["top_token"], "top_prob": r["top_prob"],
                 "topk_tokens": json.loads(r["topk_tokens"]), "topk_probs": json.loads(r["topk_probs"])}
                for r in rows]


# ----------------------------- Postgres -----------------------------
class PostgresStore:
    def __init__(self, dsn: str):
        import psycopg
        self._psycopg = psycopg
        self.dsn = dsn
        self.init_schema()

    def init_schema(self):
        import psycopg
        ddl = """
        CREATE TABLE IF NOT EXISTS infer_sessions(session_id TEXT PRIMARY KEY, prompt TEXT,
          model_id TEXT, n_layers INT, n_prompt_tok INT, n_gen_tok INT, generated_text TEXT,
          created_at TIMESTAMPTZ DEFAULT now());
        CREATE TABLE IF NOT EXISTS infer_tokens(session_id TEXT, token_pos INT, token_str TEXT,
          token_id INT, is_prompt BOOLEAN, PRIMARY KEY(session_id,token_pos));
        CREATE TABLE IF NOT EXISTS infer_layer_steps(session_id TEXT, token_pos INT, layer_idx INT,
          residual_delta DOUBLE PRECISION, hidden_norm DOUBLE PRECISION, attn_entropy DOUBLE PRECISION,
          PRIMARY KEY(session_id,token_pos,layer_idx));
        CREATE TABLE IF NOT EXISTS infer_logitlens(session_id TEXT, token_pos INT, layer_idx INT,
          top_token TEXT, top_prob REAL, topk_tokens TEXT[], topk_probs REAL[],
          PRIMARY KEY(session_id,token_pos,layer_idx));
        """
        with psycopg.connect(self.dsn) as c:
            c.execute(ddl); c.commit()

    def write_session(self, s):
        with self._psycopg.connect(self.dsn) as c, c.cursor() as cur:
            cur.execute("""INSERT INTO infer_sessions
                (session_id,prompt,model_id,n_layers,n_prompt_tok,n_gen_tok,generated_text)
                VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id) DO NOTHING""",
                (s["session_id"], s["prompt"], s["model_id"], s["n_layers"],
                 s["n_prompt"], s["n_gen"], s["gen_text"]))
            cur.executemany("INSERT INTO infer_tokens VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            [(s["session_id"], p, ts, ti, bool(b)) for (p, ts, ti, b) in s["tokens"]])
            cur.executemany("INSERT INTO infer_layer_steps VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            [(s["session_id"], p, l, d, n, e) for (p, l, d, n, e) in s["layer_rows"]])
            cur.executemany("INSERT INTO infer_logitlens VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            [(s["session_id"], p, l, tt, tp, tks, tps)
                             for (p, l, tt, tp, tks, tps) in s["lens_rows"]])
            c.commit()

    def _rows(self, sql, args):
        with self._psycopg.connect(self.dsn) as c, c.cursor() as cur:
            cur.execute(sql, args)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def list_sessions(self):
        return self._rows("""SELECT session_id,prompt,model_id,n_layers,n_prompt_tok,n_gen_tok,
            generated_text,created_at FROM infer_sessions ORDER BY created_at DESC""", ())

    def get_session(self, sid):
        m = self._rows("SELECT * FROM infer_sessions WHERE session_id=%s", (sid,))
        if not m:
            return None
        meta = m[0]
        meta["tokens"] = self._rows("SELECT token_pos,token_str,token_id,is_prompt FROM "
                                    "infer_tokens WHERE session_id=%s ORDER BY token_pos", (sid,))
        return meta

    def get_surface(self, sid, metric):
        rows = self._rows(f"SELECT layer_idx,token_pos,{metric} v FROM infer_layer_steps "
                          "WHERE session_id=%s ORDER BY layer_idx,token_pos", (sid,))
        return _surface_payload(metric, rows)

    def get_logitlens(self, sid, token_pos):
        return self._rows("""SELECT layer_idx,top_token,top_prob,topk_tokens,topk_probs
            FROM infer_logitlens WHERE session_id=%s AND token_pos=%s ORDER BY layer_idx""",
            (sid, token_pos))


def _surface_payload(metric, rows):
    def g(r, k):
        return r[k] if isinstance(r, dict) else r[k]
    pts = [{"depth_idx": g(r, "layer_idx"), "step": g(r, "token_pos"), "value": g(r, "v")} for r in rows]
    steps = sorted({p["step"] for p in pts})
    layers = [{"depth_idx": d, "layer_name": f"layer {d}", "kind": "other"}
              for d in sorted({p["depth_idx"] for p in pts})]
    return {"metric": metric, "steps": steps, "layers": layers, "points": pts}
