# From Jev to Your Laptop: Building "System One" Decision Models in ONNX with Mobius

*A developer advocate's walkthrough of ONNXRuntime Mobius, how it differs from the ONNX Runtime GenAI Model Builder, and a hands-on example that converts an open "Jev-style" decision model to CPU ONNX.*

---

## 0. Why I'm writing this now

Two things landed on my radar in the same week.

First, TypeSafe AI published **["Introducing System One Models & Jev"](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** (Sep 15, 2026) and the developer community immediately started arguing about it. The pitch is unusual: instead of another chat model, Jev is described as *"a new class of frontier models built to make fast, structured decisions that software can use directly."* The post frames Jev as **"a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out"** — and notes that while *"Jev gives up string generation, it's optimized for structured outputs and can't hallucinate."*

The numbers in that post are what got everyone's attention: end-to-end response times of **70ms–500ms** versus 3–329 seconds for frontier LLMs, input tokens at **$0.042/MTok** with output tokens free, and a claim that every answer arrives with a **calibrated probability** — *"higher confidence means higher accuracy."* The name itself is a thesis: TypeSafe named Jev after **William Stanley Jevons**, expecting that *"every order of magnitude drop in the cost of intelligence unlocks orders of magnitude more use cases."*

Second — and this is the part that matters for anyone who builds on Microsoft's AI stack — **[onnxruntime/mobius](https://github.com/onnxruntime/mobius)** has been quietly shipping at a furious pace. It's the piece of infrastructure that makes it realistic to take a model like this, in whatever format it arrives, and run it as an ONNX graph on hardware you control.

So I built the bridge. **[kinfey/JevONNX](https://github.com/kinfey/JevONNX)** takes an open, community-fine-tuned model that reproduces the System One *request shape*, converts it from GGUF to CPU ONNX with Mobius, and serves typed probabilistic decisions locally.

This post covers three things: what Mobius is, how it differs from the ONNX Runtime GenAI Model Builder you may already be using, and how the JevONNX sample puts it together.

---

## 1. What is Mobius?

**Mobius builds generative models *in* ONNX.** That preposition is the whole idea.

From the repo's own description, Mobius provides *"ONNX model definitions for GenAI using the onnxscript.nn API"* — model definitions for *"LLMs, MoE, multimodal, encoder-only, encoder-decoder, vision, audio, and diffusion models — built directly as ONNX graphs using `onnxscript.nn.Module`."*

The key sentence, and the one worth reading twice:

> *"Rather than tracing or exporting PyTorch models, it constructs the ONNX graph declaratively, then applies pretrained HuggingFace weights."*

### Why "declarative construction" is a big deal

If you've ever exported a model to ONNX the traditional way, you know the failure modes. You run `torch.onnx.export`, the tracer executes one concrete path through Python control flow, and anything dynamic — a KV cache branch, a rotary embedding computed with a Python `if`, a sliding-window mask — either silently bakes in a constant or explodes into an unreadable subgraph. You then spend a week diffing graphs.

Mobius inverts this. The graph is the **source of truth**, written as composable modules, and the weights are applied afterwards. The architecture in the README is four clean layers:

```
HuggingFace Hub
       │
       ▼
ArchitectureConfig ◄── from_transformers() / from_diffusers()
       │
       ▼
Model Module ◄── Reusable Components (Attention, MLP, RMSNorm, RoPE, …)
       │
       ▼
Task ◄── CausalLMTask, VisionLanguageTask, VAETask, DenoisingTask, …
       │
       ▼
ONNX Model ◄── preprocess_weights() + apply_weights()
```

- **Components** — `onnxscript.nn.Module` building blocks (Attention, MLP, DecoderLayer, RoPE, VisionEncoder, MoELayer…)
- **Models** — full architectures composed from those components
- **Tasks** — define the ONNX graph's I/O contract (inputs, outputs, KV cache)
- **Registry** — maps HuggingFace `model_type` strings to model classes

Adding a new architecture becomes *writing a model definition*, not *debugging an exporter*. Mobius even ships developer skills in `.agents/skills/` (`adding-a-new-model`, `reusable-components`, `moe-models`, `multimodal-models`, `writing-tests`, `writing-rewrite-rules`) so an AI coding agent can do the legwork with you.

### Coverage

This isn't a toy. The repo claims **290+ Transformers model types and 10 Diffusers component types across 40+ task types and 100+ reusable components**, spanning:

| Category | Examples |
|---|---|
| Text generation | Llama 2/3/4, Mistral, Qwen 2/2.5/3/3.5/3.6, Phi-3/3.5, Gemma 1/2/3/4, Granite, GPT-2, OPT, OLMo, SmolLM3 |
| Mixture of Experts | PhiMoE, GPTOSS, Mixtral, OLMoE, DeepSeek-V2/V3, Qwen2-MoE, Qwen3-MoE, Qwen3-Next, GLM-4-MoE, Arctic, DBRX, Jamba |
| Multimodal | Gemma 3/4, Phi-4MM (vision + audio + LoRA), LLaVA, InternVL2, MiniCPM-V 4.6, Qwen2.5-VL, Qwen3-VL, Pixtral |
| Encoder-only | BERT, RoBERTa, ALBERT, DeBERTa, DistilBERT, ELECTRA, XLNet |
| Encoder-decoder | BART, T5/mT5, Marian, M2M-100, Pegasus, BigBird-Pegasus |
| Speech-to-text | Whisper, Moonshine, Moonshine Streaming, FastConformer-RNNT, FunASR, GLM-ASR, Qwen3-ASR, SenseVoice |
| Audio | Wav2Vec2, HuBERT, WavLM, SpeechT5 |
| Vision | ViT, BEiT, DeiT, DINOv2, Swin, CLIP, SigLIP |
| Diffusion | Stable Diffusion (UNet + VAE + ControlNet), Flux, SD3, DiT, QwenImage / Qwen-Image-Edit-2509, HunyuanDiT, CogVideoX |
| Adapters | T2I-Adapter, IP-Adapter |

### The five-minute version

```bash
pip install -e .
```

```python
from mobius import build

pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama-3.2-1b/")
```

Static KV cache, when you know your max sequence length up front:

```python
from mobius import build, CausalLMTask

task = CausalLMTask(static_cache=True, max_seq_len=2048)
pkg = build("meta-llama/Llama-3.2-1B", task=task)
pkg.save("output/llama-3.2-1b-static/")
```

EP-aware optimization — the graph itself is tuned per execution provider, *"each with the right set of fused kernels and lowering passes applied automatically"*:

```python
# CUDA: GQA fusion, SkipLayerNorm, PackQKV
pkg = build("meta-llama/Llama-3.2-1B", execution_provider="cuda", dtype="f16")

# WebGPU: GQA fusion, Shape ops replaced with portable alternatives
pkg = build("meta-llama/Llama-3.2-1B", execution_provider="webgpu", dtype="f16")
```

And the CLI, which is what we'll use later:

```bash
mobius build --model Qwen/Qwen2.5-0.5B --output output_dir/
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ --ep cuda --dtype f16
mobius build --model openai/whisper-tiny --output output_dir/   # encoder/ + decoder/
```

Build-mode toggles use a cargo-style `--features` flag — available features are `static-cache`, `fp8-kv-cache`, `prune-prefill-prefix`, and `text-only`:

```bash
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ \
  --features static-cache,prune-prefill-prefix --max-seq-len 2048
```

`--release` strips build-time debug and provenance metadata to reduce saved model size, while preserving functional metadata keyed with the `mobius.` prefix.

---

## 2. Mobius vs. the ONNX Runtime GenAI Model Builder

This is the question I get every time I demo Mobius, so let's be precise. **Both are Microsoft tools, both produce ONNX, and they are not competitors — they sit at different points on the same pipeline.**

The **[ONNX Runtime GenAI Model Builder](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models)** describes itself as the tool *"for quickly creating optimized and quantized ONNX models within a few minutes that run with ONNX Runtime GenAI."* Its output is a deployment-ready package: `model.onnx` plus `genai_config.json` plus tokenizer files, wired for the `onnxruntime-genai` runtime. Its architecture list is explicitly curated — AMD OLMo, ChatGLM, DeepSeek, ERNIE 4.5, Gemma, gpt-oss, Granite, Granite MoE Hybrid, HunYuan Dense V1, InternLM2, LFM2, LFM2 MoE, Llama, Mistral, Nemotron, Phi, Qwen, SmolLM3, Whisper — and the README states plainly that *"it is intended for supporting the latest, popular state-of-the-art models."*

Mobius, by contrast, is a **model-definition library**: a component algebra for describing generative architectures as ONNX graphs, with a registry mapping HuggingFace `model_type` strings to model classes.

Here's how I'd separate them:

| Dimension | ORT GenAI Model Builder | Mobius |
|---|---|---|
| **Primary goal** | Produce a deployment-ready GenAI package, fast | Define and construct the ONNX graph itself |
| **Output contract** | `model.onnx` + `genai_config.json` + tokenizer, targeted at the ORT GenAI runtime | ONNX model package; you choose the runtime (plain ORT sessions included) |
| **How the graph is made** | Per-architecture builder subclasses (`LlamaModel`, `QwenModel`, `Phi3MiniModel`, `WhisperModel`…) overriding `make_layer` / `make_attention` / `make_moe` | Declarative `onnxscript.nn.Module` composition from reusable Components → Models → Tasks |
| **Breadth** | Curated list of ~19 popular families, text + Whisper | 290+ Transformers types, 10 Diffusers component types, 40+ tasks — including vision, audio, and diffusion |
| **Extending it** | Add a builder subclass inside the `onnxruntime-genai` tree | Add a model definition from existing components; shipped `.agents/skills/` guide it |
| **Tuning surface** | Rich `--extra_options` for quantization and runtime behavior (QMoE, paged attention, KV-cache quantization, QDQ, CUDA/WebGPU graph capture…) | `--features` build toggles + `execution_provider=` EP-aware graph lowering + `dtype` control |
| **Diffusion / multi-component pipelines** | Not the target | First-class (`--model Qwen/Qwen-Image-2512` builds all components) |
| **When I reach for it** | "I have a Phi/Llama/Qwen chat model and I want an int4 CPU/CUDA package running under ORT GenAI *today*" | "My architecture is hybrid, unusual, multimodal, diffusion, or I need graph-level control over what gets emitted" |

Notably, the two overlap on real capabilities. The Model Builder also consumes GGUF and handles Qwen3.5-class features (compact state updates, recurrent-operator selection). Mobius also emits ORT-GenAI-compatible metadata — though the repo is honest about limits: for Mage-VL, *"ORT GenAI export is currently rejected because the runtime cannot supply its required `patch_positions` input or Mage-VL's 1D decoder positions."* That single line tells you exactly where the boundary sits: **the Model Builder is bounded by what the GenAI runtime can express; Mobius is bounded by what ONNX can express.**

**My rule of thumb as an advocate:** if your model is on the Model Builder's supported list and you want the GenAI runtime's generation loop, use the Model Builder — it is faster and gives you a complete package. The moment you're off that list, or you need the graph itself to be a first-class artifact you can reason about and modify, Mobius is the tool.

The JevONNX sample is squarely in the second category. Let's look at why.

---

## 3. The sample: converting a Jev-style decision model to CPU ONNX

Repo: **[github.com/kinfey/JevONNX](https://github.com/kinfey/JevONNX)**

### 3.1 The honest framing first

Jev is TypeSafe's proprietary model. You can't download it and convert it, and this project doesn't pretend otherwise.

What the community *did* do is interesting: `n4ze3m/Qwen3.5-4B-Hmm` is **"an experimental open-model attempt to reproduce the central idea and request shape of the System One workflow introduced with Jev."** It works by *"fine-tuning Qwen3.5-4B to answer typed questions through option probabilities instead of ordinary generated text."*

The README of JevONNX is emphatic about the boundaries, and I want to repeat them here rather than bury them:

> *"Hmm is not Jev, is not affiliated with TypeSafe AI, and does not reproduce Jev's proprietary model architecture, parallel sampler, performance, calibration, or type-safety guarantees. The Hmm model card explicitly states that it is less capable than Jev."*

And: *"The reproduction is behavioral rather than architectural."*

The distinction, straight from the project's comparison table:

| Property | Jev | Hmm in this project |
|---|---|---|
| Model design | Native System One Model | Qwen3.5-4B fine-tuned to reproduce the typed-decision concept |
| Output mechanism | Native typed probabilistic outputs | First-token option-letter probabilities |
| Multiple questions | Designed for parallel sampling | Jev-like request shape, implemented as one model pass per question |
| Output types | Schema-constrained structured values | `noul`, `choice`, and `score` adapter |
| Runtime here | TypeSafe service | Local ONNX Runtime CPU |
| Guarantees | As documented by TypeSafe | No equivalent guarantee is claimed |

The mechanism is refreshingly legible: the adapter turns each typed question into an exact multiple-choice prompt, then *"reads the first generated-token logits for A, B, C, and subsequent option letters and normalizes them into probabilities."* No JSON parsing, no retry-on-malformed-output, no regex. The probability distribution *is* the answer.

### 3.2 Why this needs Mobius specifically

Qwen3.5 is a **hybrid** architecture, and that's the whole reason this is a Mobius story.

The exported model has **32 hybrid decoder layers**. Linear-attention layers carry `conv_state` and `recurrent_state`; full-attention layers carry key/value caches. In this artifact, full-attention layers sit at indices **3, 7, 11, 15, 19, 23, 27, and 31**.

The practical consequence, quoted from the README:

> *"The prompt is processed one token at a time while every `present.*` output is fed back into its corresponding `past_key_values.*` input. This differs from a conventional decoder-only example that can prefill an entire prompt in one call."*

That is a graph-shape problem. You cannot handwave it with a config flag — you need the exported graph's state contract to be explicit and inspectable, which is exactly what declarative construction gives you. The implementation *"follows the structure of Mobius `examples/text_generation.py` while adding the hybrid-state handling required by Qwen3.5."*

### 3.3 The conversion pipeline

```
n4ze3m/Qwen3.5-4B-Hmm-Q4_K_M.gguf
        │  SHA-256 verification
        ▼
Mobius GGUF reader/importer
        │  metadata and tensor mapping
        │  Qwen3.5 hybrid graph construction
        │  Q4_K_M → packed quantized target storage
        │  CPU execution-provider optimization
        ▼
onnx_outputs/
 ├── model.onnx
 ├── model.onnx.data
 ├── quantization_report.json
 ├── tokenizer.json
 ├── tokenizer_config.json
 ├── chat_template.jinja
 └── cpu_test_summary.json
        │
        ▼
ONNX Runtime CPUExecutionProvider
        │  token-by-token hybrid-state prefill
        │  first-token A/B/C/… probabilities
        ▼
typed decision JSON response
```

One command does the conversion:

```bash
mobius build-gguf Qwen3.5-4B-Hmm-Q4_K_M.gguf \
  --output onnx_outputs \
  --ep cpu \
  --dtype f32 \
  --release
```

Two details worth internalizing:

- **`--dequantize` is intentionally not used.** Keeping quantized storage *"substantially reduc[es] the output size compared with a fully dequantized FP32 model."* The external-data file is still ~2.6 GB — imagine it dequantized.
- **Don't trust the preset, read the report.** *"Q4_K conversion can involve a lossy affine repack; inspect `quantization_report.json` rather than assuming source-preset fidelity."* This is the kind of line that separates a demo from a deployment.

### 3.4 Reproducibility — the part I actually want you to copy

Every pin, in a table, in the README:

| Component | Revision |
|---|---|
| GGUF repository | `n4ze3m/Qwen3.5-4B-Hmm` |
| GGUF revision | `c27fa3c627dfaced343c6ba9d3a0d00243a3be51` |
| GGUF filename | `Qwen3.5-4B-Hmm-Q4_K_M.gguf` |
| GGUF SHA-256 | `5e03cb057049c56b421bd3c506d77fd8e0a77996bb8464148a02cfc3caac5229` |
| Mobius revision | `6b27a3f08b8b5d08ba9b14b416e3b435942bb0bd` |
| Tokenizer repository | `Qwen/Qwen3.5-4B` |
| Tokenizer revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

The notebook **builds a Mobius wheel from the pinned Git source** rather than installing a prebuilt package. On a project moving at 500+ commits, that is not paranoia — that is the only way your conversion is still reproducible next month.

### 3.5 Running it

Two files carry the project: `Qwen3_5_4B_Hmm_GGUF_to_CPU_ONNX_Mobius_fixed.ipynb` (the Colab conversion and validation workflow) and `run_hmm_onnx.py` (the standalone typed-decision runner). Run the notebook in a high-memory Colab CPU runtime; it writes to `/content/onnx_outputs` on Colab and `.mobius_colab_run/onnx_outputs` elsewhere.

Then:

```bash
python run_hmm_onnx.py --model-dir .mobius_colab_run/onnx_outputs
python run_hmm_onnx.py --model-dir .mobius_colab_run/onnx_outputs --request request.json
```

A request is state plus typed questions:

```json
{
  "state": "Help! My payouts have failed for 3 days. I need the money today.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this message convey urgency?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "priority": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": ["Not urgent", "Normal", "Urgent", "Critical"]
    }
  }
}
```

From Python:

```python
from pathlib import Path
from run_hmm_onnx import HmmOnnx

model = HmmOnnx(Path(".mobius_colab_run/onnx_outputs"))
response = model.decide({
    "state": "Help! My payouts have failed for 3 days.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
        }
    },
})
print(response)
```

### 3.6 The three decision types

**`noul`** — a nullable-style yes/no probability; the probability of the `true` option:

```json
{ "type": "noul", "noul": 0.9745 }
```

**`choice`** — the highest-probability key, all normalized option probabilities, and a confidence value measuring how peaked the distribution is:

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 0.7216, "technical": 0.2759, "sales": 0.0025 },
  "confidence": 0.5823
}
```

**`score`** — the expected numeric level, the level legend, normalized probabilities, and confidence:

```json
{
  "type": "score",
  "score": 2.6653,
  "legend": { "0": "Not urgent", "1": "Normal", "2": "Urgent", "3": "Critical" },
  "probabilities": { "0": 0.0061, "1": 0.0361, "2": 0.2443, "3": 0.7135 },
  "confidence": 0.618
}
```

Look at that `score` output for a second. You don't get "this is critical." You get `2.6653` — a distribution leaning hard on *Critical* but with real mass on *Urgent*. That's a number your routing code can threshold on, log, and A/B test. It's the difference between an LLM that *says* it's confident and a distribution you can actually calibrate against.

One constraint: **Hmm is trained with 26 option letters.** Larger option sets are handled by splitting them into groups of 25 and adding a `none_of_these` option to each group, matching the reference Hmm server strategy.

### 3.7 Does it actually run?

Yes — the README reports the generated model executed directly on an **Apple Silicon CPU with ONNX Runtime**. A 4B hybrid-architecture model, converted from GGUF, running typed decisions on a laptop CPU with no service dependency.

---

## 4. What I'd tell a room of developers

**The System One idea is portable even where the model isn't.** You can't run Jev locally, but the *shape* — state in, typed decisions with probabilities out — is something you can prototype today with an open model and an ONNX graph. As TypeSafe puts it, structured outputs *"slot into ordinary software as fuzzy decision rules: classify, route, score, extract, or branch where hand-written logic is too brittle."* That's a pattern, not a product.

**Probabilities are the actual product.** The reason this class of model matters isn't speed alone. TypeSafe's argument is sharp: *"If a model can do a task 95% of the time but doesn't say when it's in the 5%, it can't automate that task."* Every engineer who has shipped an LLM feature has felt that exact wall.

**Set expectations honestly.** JevONNX is an educational reproduction. It gives you `noul`/`choice`/`score` on a local CPU. It does not give you Jev's calibration, parallel sampler, or type-safety guarantees — and the project says so in its own README. Ship that caveat with your demo.

**Mobius is the infrastructure bet worth learning.** When a new architecture shows up — hybrid attention, a novel MoE routing scheme, a three-model multimodal export — the question stops being "will the exporter support it?" and becomes "which components do I compose?" That is a much better question to be asking.

---

## Links

- Mobius — https://github.com/onnxruntime/mobius (docs: https://onnxruntime.github.io/mobius/)
- ONNX Runtime GenAI Model Builder — https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models
- JevONNX sample — https://github.com/kinfey/JevONNX
- Introducing System One Models & Jev — https://typesafe.ai/blog/introducing-system-one-models-and-jev

*Mobius is MIT-licensed. JevONNX is an independent experiment and is not affiliated with or endorsed by TypeSafe AI.*
