# VisPrompter

Prompt **any** HuggingFace language model and visualize, **layer by layer**, how the
answer forms. Ask a question on the left, watch the per-layer contribution surface
and the logit-lens replay on the right. Standalone — no Postgres or docker required.

## Install

```bash
pip install -e ".[serve]"        # add ,pg for an optional Postgres store
# torch + transformers are deps; install the CUDA torch build that fits your GPU
```

## Use it (three ways)

**CLI — capture, then view:**
```bash
visprompter run  --model google/gemma-4-E4B-it --prompt "Why is the sky blue?" --store ./vp.db
visprompter ui   --store ./vp.db --port 9000          # open http://localhost:9000
```

**CLI — live server (load once, prompt from the web UI):**
```bash
visprompter serve --model google/gemma-4-E4B-it --port 9000
```

**Library:**
```python
from visprompter import VisPrompterCapture
cap = VisPrompterCapture.from_pretrained("google/gemma-4-E4B-it", store="./vp.db")
cap.capture("Why is the sky blue?", max_new_tokens=32)
# then:  visprompter ui --store ./vp.db --port 9000
```

## Use a locally stored model

`--model` (and `from_pretrained`'s first arg) is passed straight to
`transformers.from_pretrained`, so a **local directory works exactly like a Hub id**
— no download, no HF token needed. Point it at any folder that contains the usual
files (`config.json`, tokenizer files, and `*.safetensors`/`*.bin` weights):

```bash
# a fine-tune / checkpoint saved with model.save_pretrained(dir) + tokenizer.save_pretrained(dir)
visprompter serve --model /data/models/my-gemma-finetune --port 9000

# or capture from CLI
visprompter run --model ./checkpoints/step-2000 --prompt "Summarize this." --store ./vp.db
visprompter ui  --store ./vp.db --port 9000
```

Library form:
```python
from visprompter import VisPrompterCapture
cap = VisPrompterCapture.from_pretrained("/data/models/my-gemma-finetune", store="./vp.db")
cap.capture("Summarize this.", max_new_tokens=64)
```

Or wrap a model you've **already loaded** (the most flexible path — quantized,
custom `device_map`, `trust_remote_code`, a PEFT-merged model, etc.):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from visprompter import VisPrompterCapture

path = "/data/models/my-gemma-finetune"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map="cuda:0")
cap = VisPrompterCapture(model, tok, model_id=path, store="./vp.db")
cap.capture("Why is the sky blue?", max_new_tokens=32)
```

Notes:
- The directory must be a full model (config + tokenizer + weights). A bare LoRA
  adapter folder won't load on its own — merge it into the base first
  (`peft … merge_and_unload()` then `save_pretrained`) or load base+adapter
  yourself and pass the live model to `VisPrompterCapture(model, tok, …)`.
- Fully offline? `export HF_HUB_OFFLINE=1` so transformers never reaches the Hub.
- Models needing custom code: load the model yourself with
  `trust_remote_code=True` and pass it in via `VisPrompterCapture(model, tok, …)`.

## The UI

A resizable split (drag any divider):
- **Left** — **input** (prompt + Run) on top, the generated **answer** below
  (click any token to inspect it).
- **Right** — the **3D visualizer** (layer depth × token position; height/color =
  the chosen signal, with token-by-token replay) on top, and the **logit-lens**
  "answer forming" panel below.

## Signals captured (per token, per layer)

| signal | meaning |
|---|---|
| `residual_delta` `‖h_l − h_{l−1}‖` | how much that layer moved the residual stream |
| `hidden_norm` `‖h_l‖` | hidden-state magnitude |
| `attn_entropy` | attention focus/diffuseness |
| logit lens | each layer's hidden state decoded to the vocab — the answer forming |

## Model-agnostic

Layer count from `output_hidden_states`; lm_head from `get_output_embeddings()`; the
final norm discovered by name **and matched to the lm_head's hidden dim** (so it
won't grab a wrong-sized sub-norm in matformer/multimodal models). Works on
Llama/Qwen/Gemma/GPT-2/etc.

## Storage — there's no database server

VisPrompter captures per-(token, layer) signals during inference and the UI
*queries them back* (the 3D surface, logit lens, and replay all read stored data).
The "store" is just that hand-off between the model and the browser — **never a
separate server you have to run**. Three options, easiest first:

| Store | What it is | When |
|---|---|---|
| **in-memory** | pure RAM, **no file** | live `serve` you don't need to keep |
| **SQLite** (default) | one local `.db` file, **auto-created** | keep sessions to revisit/compare |
| Postgres | `postgresql://…` DSN | share with a larger stack |

```bash
# nothing on disk — everything stays in RAM:
visprompter serve --model <id> --port 9000 --ephemeral

# default: auto-creates ./visprompter.db (parent dirs made for you):
visprompter serve --model <id> --port 9000
visprompter serve --model <id> --port 9000 --store ~/vp/runs.db

# library — same choices:
VisPrompterCapture.from_pretrained("<id>", store=None)            # in-memory
VisPrompterCapture.from_pretrained("<id>", store="./vp.db")       # SQLite file
VisPrompterCapture.from_pretrained("<id>", store="postgresql://…")
```

SQLite is built into Python, so the package is fully self-contained: `pip install`,
run, done — the `.db` file appears on its own (or use `--ephemeral` for none).
