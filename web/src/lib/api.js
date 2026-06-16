const BASE = import.meta.env.VITE_API_URL || "";

async function get(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

export const api = {
  infer: async (prompt, maxNew = 32) => {
    const r = await fetch(`${BASE}/api/infer`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_new_tokens: maxNew }),
    });
    if (!r.ok) throw new Error((await r.text()) || `infer ${r.status}`);
    return r.json();
  },
  sessions: () => get("/api/sessions"),
  session: (id) => get(`/api/sessions/${id}`),
  surface: (id, metric) => get(`/api/sessions/${id}/surface?metric=${metric}`),
  logitlens: (id, tokenPos) => get(`/api/sessions/${id}/logitlens?token_pos=${tokenPos}`),
};
