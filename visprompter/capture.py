"""VisPrompterCapture — run a query through any HF LM and capture, per (token, layer):
residual contribution, hidden norm, attention entropy, and a logit-lens top-k.

Backend-agnostic (writes to any visprompter store) and model-agnostic.
"""
from __future__ import annotations

import uuid
from typing import Optional

import torch
import torch.nn.functional as F

from .discover import discover_final_norm, discover_lm_head
from .store import open_store


class VisPrompterCapture:
    def __init__(self, model, tokenizer, model_id: str = "model",
                 store=None, topk: int = 5):
        self.model = model.eval()
        self.tok = tokenizer
        self.model_id = model_id
        self.topk = topk
        self.store = store if (store is None or hasattr(store, "write_session")) else open_store(store)
        self.lm_head = discover_lm_head(model)
        self.final_norm = discover_final_norm(model, lm_head=self.lm_head)

    @classmethod
    def from_pretrained(cls, model_id: str, store=None, dtype="auto",
                        device_map="auto", hf_token: Optional[str] = None,
                        auto_class: Optional[str] = None, topk: int = 5, **kw):
        from transformers import AutoTokenizer
        import transformers
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        # pick a sensible auto class: text LMs -> CausalLM; multimodal -> ImageTextToText
        Cls = getattr(transformers, auto_class) if auto_class else None
        if Cls is None:
            try:
                Cls = transformers.AutoModelForCausalLM
                model = Cls.from_pretrained(model_id, token=hf_token, dtype=dtype,
                                            device_map=device_map, **kw)
            except Exception:
                Cls = transformers.AutoModelForImageTextToText
                model = Cls.from_pretrained(model_id, token=hf_token, dtype=dtype,
                                            device_map=device_map, **kw)
        else:
            model = Cls.from_pretrained(model_id, token=hf_token, dtype=dtype,
                                        device_map=device_map, **kw)
        return cls(model, tok, model_id=model_id, store=store, topk=topk)

    def _encode(self, prompt: str):
        dev = next(self.model.parameters()).device
        if getattr(self.tok, "chat_template", None):
            enc = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
        else:
            enc = self.tok(prompt, return_tensors="pt").to(dev)
        return enc

    @torch.no_grad()
    def _logit_lens(self, hidden):
        h = hidden
        if self.final_norm is not None:
            try:
                h = self.final_norm(hidden)
            except Exception:
                self.final_norm = None     # disable on first failure; lens still works
                h = hidden
        logits = self.lm_head(h)
        probs = F.softmax(logits.float(), dim=-1)
        return probs.topk(self.topk, dim=-1)   # (vals, idx) each [seq, k]

    @torch.no_grad()
    def capture(self, prompt: str, max_new_tokens: int = 32, persist: bool = True):
        model, tok = self.model, self.tok
        enc = self._encode(prompt)
        ids = enc["input_ids"]
        n_prompt = ids.shape[1]

        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             return_dict_in_generate=True)
        seq = gen.sequences[0]
        full_len = seq.shape[0]
        gen_text = tok.decode(seq[n_prompt:], skip_special_tokens=True)

        out = model(input_ids=seq.unsqueeze(0), output_hidden_states=True,
                    output_attentions=True, use_cache=False)
        hs = out.hidden_states
        attns = out.attentions
        n_layers = len(hs) - 1
        seq_len = hs[0].shape[1]

        tokens = [(pos, tok.decode([int(seq[pos])]), int(seq[pos]), pos < n_prompt)
                  for pos in range(full_len)]

        layer_rows, lens_rows = [], []
        for l in range(1, n_layers + 1):
            h_cur, h_prev = hs[l][0], hs[l - 1][0]
            delta = torch.linalg.vector_norm(h_cur - h_prev, dim=-1)
            hnorm = torch.linalg.vector_norm(h_cur, dim=-1)
            if attns is not None and l - 1 < len(attns) and attns[l - 1] is not None:
                a = attns[l - 1][0].float().clamp_min(1e-12)
                ent = (-(a * a.log()).sum(-1)).mean(0)
            else:
                ent = torch.zeros(seq_len, device=h_cur.device)
            tk_p, tk_i = self._logit_lens(h_cur)
            d_c, n_c, e_c = delta.float().cpu().tolist(), hnorm.float().cpu().tolist(), ent.float().cpu().tolist()
            for pos in range(seq_len):
                layer_rows.append((pos, l - 1, d_c[pos], n_c[pos], e_c[pos]))
                ids_k = [int(x) for x in tk_i[pos].tolist()]
                probs_k = [float(x) for x in tk_p[pos].tolist()]
                toks_k = [tok.decode([i]) for i in ids_k]
                lens_rows.append((pos, l - 1, toks_k[0], probs_k[0], toks_k, probs_k))

        session = dict(session_id=str(uuid.uuid4()), prompt=prompt, model_id=self.model_id,
                       n_layers=n_layers, n_prompt=n_prompt, n_gen=full_len - n_prompt,
                       gen_text=gen_text, tokens=tokens, layer_rows=layer_rows, lens_rows=lens_rows)
        if persist and self.store is not None:
            self.store.write_session(session)
        return session


# back-compat alias
OmniVisCapture = VisPrompterCapture
