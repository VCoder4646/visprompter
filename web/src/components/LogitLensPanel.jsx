import React, { useEffect, useRef, useState } from "react";
import { api } from "../lib/api.js";

const FORM_MS = 220;

// Logit-lens "answer forming": for a selected token, decode every layer's hidden
// state to its predicted next-token. Play sweeps a depth playhead so you watch the
// prediction crystallize layer by layer.
export default function LogitLensPanel({ sessionId, tokenPos, tokens }) {
  const [lens, setLens] = useState([]);
  const [formLayer, setFormLayer] = useState(null);
  const [playing, setPlaying] = useState(false);
  const idxRef = useRef(0);

  useEffect(() => {
    if (sessionId == null || tokenPos == null) { setLens([]); return; }
    setFormLayer(null); setPlaying(false);
    api.logitlens(sessionId, tokenPos).then(setLens).catch(() => setLens([]));
  }, [sessionId, tokenPos]);

  useEffect(() => {
    if (!playing || !lens.length) return;
    const id = setInterval(() => {
      idxRef.current += 1;
      if (idxRef.current >= lens.length) { idxRef.current = 0; }
      setFormLayer(lens[idxRef.current].layer_idx);
    }, FORM_MS);
    return () => clearInterval(id);
  }, [playing, lens]);

  if (tokenPos == null)
    return <div style={{ fontSize: 13, opacity: 0.55 }}>
      Click a token (in the answer strip or a bar in the surface) to see how its
      prediction forms across layers.
    </div>;

  if (!lens.length)
    return <div style={{ fontSize: 13, opacity: 0.55 }}>loading logit lens…</div>;

  const finalTok = lens[lens.length - 1]?.top_token;
  const tokStr = tokens?.find((t) => t.token_pos === tokenPos)?.token_str;
  // depth where the final prediction first locks in
  let lockIdx = -1;
  for (let i = 0; i < lens.length; i++) if (lens[i].top_token === finalTok) { lockIdx = i; break; }

  return (
    <div style={{ fontSize: 13 }}>
      <div style={{ fontWeight: 600 }}>logit lens · token {tokenPos}</div>
      {tokStr != null && <div style={{ opacity: 0.7, marginBottom: 4 }}>"{tokStr}"</div>}
      <div style={{ opacity: 0.7, marginBottom: 6 }}>
        final prediction: <b style={{ color: "#7fd1ff" }}>{q(finalTok)}</b>
        {lockIdx >= 0 && <span style={{ opacity: 0.6 }}> · locks in at layer {lens[lockIdx].layer_idx}</span>}
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button onClick={() => { if (!playing) { idxRef.current = 0; setFormLayer(lens[0].layer_idx); } setPlaying((p) => !p); }}
                style={pbtn} title="sweep depth — watch the answer form">{playing ? "⏸" : "▶ form"}</button>
        <button onClick={() => { setPlaying(false); setFormLayer(null); }} style={pbtn}>⤢</button>
      </div>

      <div>
        {lens.map((l) => {
          const isFinal = l.top_token === finalTok;
          const active = formLayer != null && l.layer_idx === formLayer;
          const dim = formLayer != null && l.layer_idx > formLayer;
          return (
            <div key={l.layer_idx}
                 style={{ display: "flex", alignItems: "center", gap: 6, padding: "2px 4px",
                          opacity: dim ? 0.28 : 1,
                          background: active ? "#243b5a" : "transparent", borderRadius: 4 }}>
              <span style={{ width: 30, textAlign: "right", opacity: 0.5, fontSize: 11 }}>{l.layer_idx}</span>
              <span style={{ width: 90, color: isFinal ? "#7fffa0" : "#e6e6e6",
                             overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {q(l.top_token)}
              </span>
              <div style={{ flex: 1, height: 8, background: "#0e131c", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${Math.round((l.top_prob || 0) * 100)}%`, height: "100%",
                              background: isFinal ? "#3fbf6f" : "#3b82f6" }} />
              </div>
              <span style={{ width: 34, textAlign: "right", opacity: 0.6, fontSize: 11 }}>
                {Math.round((l.top_prob || 0) * 100)}%
              </span>
            </div>
          );
        })}
      </div>
      <div style={{ opacity: 0.5, fontSize: 11, marginTop: 8 }}>
        green = matches the final answer · bar = the layer's confidence in its top token
      </div>
    </div>
  );
}

const q = (s) => s == null ? "—" : JSON.stringify(s).slice(1, -1);
const pbtn = { borderRadius: 8, border: "1px solid #2a3344", background: "#141a26",
               color: "#e6e6e6", cursor: "pointer", padding: "5px 10px", fontSize: 12 };
