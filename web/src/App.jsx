import React, { useEffect, useRef, useState } from "react";
import { api } from "./lib/api.js";
import Surface3D from "./components/Surface3D.jsx";
import LogitLensPanel from "./components/LogitLensPanel.jsx";
import Split from "./components/Split.jsx";

const METRICS = [
  ["residual_delta", "residual Δ (layer contribution)"],
  ["hidden_norm", "hidden norm"],
  ["attn_entropy", "attention entropy"],
];
const TOK_MS = 320;

export default function App() {
  const [prompt, setPrompt] = useState("Why is the sky blue?");
  const [maxNew, setMaxNew] = useState(32);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [session, setSession] = useState(null);
  const [metric, setMetric] = useState("residual_delta");
  const [logScale, setLogScale] = useState(false);

  const [layers, setLayers] = useState([]);
  const [points, setPoints] = useState([]);
  const [revealTok, setRevealTok] = useState(null);
  const [selTok, setSelTok] = useState(null);
  const [playing, setPlaying] = useState(false);
  const idxRef = useRef(0);

  useEffect(() => { refreshSessions(); }, []);
  const refreshSessions = () =>
    api.sessions().then((ss) => { setSessions(ss); if (ss.length && !sessionId) setSessionId(ss[0].session_id); })
      .catch(() => {});

  useEffect(() => {
    if (!sessionId) return;
    setPoints([]); setLayers([]); setRevealTok(null); setSelTok(null); setPlaying(false);
    api.session(sessionId).then(setSession).catch(() => setSession(null));
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    api.surface(sessionId, metric).then((s) => { setLayers(s.layers || []); setPoints(s.points || []); })
      .catch(() => { setLayers([]); setPoints([]); });
  }, [sessionId, metric]);

  const maxTok = session ? (session.n_prompt_tok + session.n_gen_tok - 1) : 0;
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      idxRef.current += 1;
      if (idxRef.current > maxTok) idxRef.current = 0;
      setRevealTok(idxRef.current);
    }, TOK_MS);
    return () => clearInterval(id);
  }, [playing, maxTok]);

  const run = async () => {
    setRunning(true); setError(null);
    try {
      const res = await api.infer(prompt, maxNew);
      await refreshSessions();
      setSessionId(res.session_id);
    } catch (e) { setError(String(e.message || e)); }
    finally { setRunning(false); }
  };

  const resetKey = `${sessionId}|${metric}`;

  // ---- panes ----
  const inputPane = (
    <div style={{ padding: 14, height: "100%", overflow: "auto" }}>
      <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>Prompt</div>
      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)}
                placeholder="ask the model something…"
                style={{ ...input, width: "100%", height: 90, resize: "vertical" }}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !running) run(); }} />
      <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 10 }}>
        <button onClick={run} disabled={running} style={runBtn}>
          {running ? "running…" : "▶ Run inference"}
        </button>
        <label style={{ fontSize: 12, opacity: 0.7 }}>tokens</label>
        <input type="number" min={1} max={128} value={maxNew}
               onChange={(e) => setMaxNew(+e.target.value)} style={{ ...input, width: 70 }} />
        <span style={{ fontSize: 11, opacity: 0.5 }}>⌘/Ctrl+Enter</span>
      </div>
      {error && <div style={{ color: "#ff8080", fontSize: 12, marginTop: 8 }}>{error}</div>}

      <div style={{ fontSize: 12, opacity: 0.7, margin: "14px 0 4px" }}>Session</div>
      <select value={sessionId || ""} onChange={(e) => setSessionId(e.target.value)}
              style={{ ...input, width: "100%" }}>
        {sessions.map((s) => (
          <option key={s.session_id} value={s.session_id}>
            {s.prompt.slice(0, 50)}{s.prompt.length > 50 ? "…" : ""}
          </option>
        ))}
      </select>
      {!sessions.length && <div style={{ fontSize: 12, opacity: 0.5, marginTop: 8 }}>
        No sessions yet — run a prompt above.</div>}
    </div>
  );

  const outputPane = (
    <div style={{ height: "100%", minHeight: 0, overflow: "auto", padding: "12px 14px" }}>
      <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>Answer (click a token)</div>
      {session ? (
        <div style={{ fontSize: 14, lineHeight: 1.9 }}>
          {session.tokens.map((t) => (
            <span key={t.token_pos} onClick={() => setSelTok(t.token_pos)}
                  title={`pos ${t.token_pos}${t.is_prompt ? " (prompt)" : ""}`}
                  style={{
                    cursor: "pointer", padding: "1px 2px", borderRadius: 3, whiteSpace: "pre-wrap",
                    background: selTok === t.token_pos ? "#3b82f6"
                      : (revealTok != null && t.token_pos === revealTok ? "#7a5cff" : "transparent"),
                    color: t.is_prompt ? "#7686a0" : "#e6e6e6",
                  }}>{t.token_str}</span>
          ))}
        </div>
      ) : <div style={{ opacity: 0.5, fontSize: 13 }}>run a prompt to see the answer</div>}
    </div>
  );

  // logit lens now lives on the RIGHT, under the 3D surface
  const lensPane = (
    <div style={{ height: "100%", minHeight: 0, overflow: "auto", padding: "12px 14px" }}>
      <LogitLensPanel sessionId={sessionId} tokenPos={selTok} tokens={session?.tokens} />
    </div>
  );

  const vizPane = (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
        <div style={{ position: "absolute", top: 10, left: 12, zIndex: 2, display: "flex", gap: 8 }}>
          <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ ...input, fontSize: 12 }}>
            {METRICS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
          </select>
          <label style={{ fontSize: 12, opacity: 0.8, display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} /> log
          </label>
        </div>
        {points.length ? (
          <Surface3D layers={layers} points={points} metric={metric} logScale={logScale}
                     resetKey={resetKey} revealStep={revealTok} stepLabel="token"
                     onSelect={(id) => setSelTok(id.step)} onRange={() => {}} highlightSteps={null} />
        ) : (
          <div style={{ padding: 24, opacity: 0.6 }}>
            {sessionId ? "loading…" : "Run a prompt to visualize how each layer contributes."}
          </div>
        )}
        <div style={{ position: "absolute", bottom: 8, left: 12, fontSize: 11, opacity: 0.55 }}>
          x = layer depth · z = token position · height/color = {metric}
        </div>
      </div>
      {session && (
        <div style={playbar}>
          <button onClick={() => { if (!playing) { idxRef.current = 0; setRevealTok(0); } setPlaying((p) => !p); }}
                  style={pbtn} title="replay generation token by token">{playing ? "⏸" : "▶"}</button>
          <button onClick={() => { setPlaying(false); setRevealTok(null); }} style={pbtn} title="show full">⤢</button>
          <input type="range" min={0} max={maxTok} value={revealTok ?? maxTok}
                 onChange={(e) => { setPlaying(false); setRevealTok(+e.target.value); }} style={{ flex: 1 }} />
          <div style={{ width: 160, fontSize: 12, opacity: 0.8 }}>
            {revealTok != null
              ? `token ${revealTok}/${maxTok}${session.tokens[revealTok] ? ` · "${session.tokens[revealTok].token_str.trim()}"` : ""}`
              : "full sequence"}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <header style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 14px",
                       borderBottom: "1px solid #1c2433" }}>
        <div style={{ fontSize: 15, fontWeight: 700 }}>VisPrompter</div>
        <div style={{ fontSize: 12, opacity: 0.55 }}>prompt a model · watch the answer form, layer by layer</div>
      </header>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Split direction="h" initial={34}>
          <Split direction="v" initial={42}>
            {inputPane}
            {outputPane}
          </Split>
          <Split direction="v" initial={64}>
            {vizPane}
            {lensPane}
          </Split>
        </Split>
      </div>
    </div>
  );
}

const input = { background: "#0e131c", color: "#e6e6e6", border: "1px solid #1c2433",
                borderRadius: 6, padding: "7px 9px", fontSize: 13 };
const runBtn = { background: "#1d4ed8", color: "#fff", border: "none", borderRadius: 6,
                 padding: "8px 14px", cursor: "pointer", fontSize: 13, whiteSpace: "nowrap" };
const playbar = { display: "flex", alignItems: "center", gap: 10, padding: "8px 14px",
                  borderTop: "1px solid #1c2433", background: "#0b0e14" };
const pbtn = { width: 34, height: 34, borderRadius: 8, border: "1px solid #2a3344",
               background: "#141a26", color: "#e6e6e6", cursor: "pointer" };
