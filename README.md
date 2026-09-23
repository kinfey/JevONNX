# Qwen3.5-4B-Hmm GGUF to CPU ONNX

[简体中文](./README.zh.md)

This project converts
[`n4ze3m/Qwen3.5-4B-Hmm`](https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm)
from GGUF to a CPU-oriented ONNX model with
[Microsoft ONNXRuntime Mobius](https://github.com/onnxruntime/mobius), then exposes
the model through its typed probabilistic decision interface. Hmm is an experimental
open-model attempt to reproduce the central idea and request shape of the System One
workflow introduced with Jev in TypeSafe AI's
[Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Hmm reproduces that idea by fine-tuning Qwen3.5-4B to answer typed questions through
option probabilities instead of ordinary generated text. It remains an independent
experiment: Hmm is not Jev, is not affiliated with TypeSafe AI, and does not reproduce
Jev's proprietary model architecture, parallel sampler, performance, calibration, or
type-safety guarantees. The Hmm model card explicitly states that it is less capable
than Jev.

## What this project provides

- A Colab-compatible notebook that builds Mobius from source.
- Reproducible download and SHA-256 verification of the requested GGUF artifact.
- GGUF-to-ONNX conversion targeting ONNX Runtime CPU execution.
- Preserved quantized weight storage where supported by Mobius.
- ONNX graph validation and direct `CPUExecutionProvider` execution.
- Stateful inference for Qwen3.5's hybrid DeltaNet and full-attention architecture.
- A Python API compatible with Hmm's `noul`, `choice`, and `score` decision shapes.
- A standalone runner for the converted ONNX model.
- A separate Qwen3.5-0.8B fine-tuning and CUDA FP16 ONNX export notebook using Microsoft Olive.

## System One, Jev, and Hmm

TypeSafe describes a System One Model as a model designed for fast, structured
decisions that software can consume directly: unstructured or structured state goes
in, and typed values with probabilities come out. Jev is TypeSafe's native System One
Model and is designed around parallel structured outputs rather than free-form string
generation.

Hmm is a fine-tuned Qwen3.5-4B experiment intended to reproduce Jev's core System One
idea in an open model: provide state, ask typed questions, and receive decisions with
probabilities rather than prose. Its local server deliberately follows a Jev-compatible
request shape to make the concept easy to evaluate.

The reproduction is behavioral rather than architectural. Hmm converts each question
into an exact multiple-choice prompt. It then reads the first generated-token logits
for `A`, `B`, `C`, and subsequent option letters and normalizes them into probabilities.

The distinction matters:

| Property | Jev | Hmm in this project |
|---|---|---|
| Model design | Native System One Model | Qwen3.5-4B fine-tuned to reproduce the typed-decision concept |
| Output mechanism | Native typed probabilistic outputs | First-token option-letter probabilities |
| Multiple questions | Designed for parallel sampling | Jev-like request shape, implemented as one model pass per question |
| Output types | Schema-constrained structured values | `noul`, `choice`, and `score` adapter |
| Runtime here | TypeSafe service | Local ONNX Runtime CPU |
| Guarantees | As documented by TypeSafe | No equivalent guarantee is claimed |

## Conversion architecture

```text
n4ze3m/Qwen3.5-4B-Hmm-Q4_K_M.gguf
                  |
                  | SHA-256 verification
                  v
        Mobius GGUF reader/importer
                  |
                  | metadata and tensor mapping
                  | Qwen3.5 hybrid graph construction
                  | Q4_K_M -> packed quantized target storage
                  | CPU execution-provider optimization
                  v
            onnx_outputs/
            ├── model.onnx
            ├── model.onnx.data
            ├── quantization_report.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── chat_template.jinja
            └── cpu_test_summary.json
                  |
                  v
       ONNX Runtime CPUExecutionProvider
                  |
                  | token-by-token hybrid-state prefill
                  | first-token A/B/C/... probabilities
                  v
       typed decision JSON response
```

The conversion command is:

```bash
mobius build-gguf Qwen3.5-4B-Hmm-Q4_K_M.gguf \
  --output onnx_outputs \
  --ep cpu \
  --dtype f32 \
  --release
```

`--dequantize` is intentionally not used. Mobius keeps supported tensors in packed
quantized storage, substantially reducing the output size compared with a fully
dequantized FP32 model. Q4_K conversion can involve a lossy affine repack; inspect
`quantization_report.json` rather than assuming source-preset fidelity.

## Qwen3.5 runtime state

The exported model has 32 hybrid decoder layers:

- Linear-attention layers carry `conv_state` and `recurrent_state`.
- Full-attention layers carry key/value caches.
- In this artifact, full-attention layers occur at indices 3, 7, 11, 15, 19, 23,
  27, and 31.
- The prompt is processed one token at a time while every `present.*` output is fed
  back into its corresponding `past_key_values.*` input.

This differs from a conventional decoder-only example that can prefill an entire
prompt in one call. The implementation follows the structure of Mobius
[`examples/text_generation.py`](https://github.com/onnxruntime/mobius/blob/main/examples/text_generation.py)
while adding the hybrid-state handling required by Qwen3.5.

## Files

| File | Purpose |
|---|---|
| `Qwen3_5_4B_Hmm_GGUF_to_CPU_ONNX_Mobius_fixed.ipynb` | Colab conversion and validation workflow |
| `run_hmm_onnx.py` | Standalone typed-decision runner |
| [`finetuning/Qwen3_5_0.8B_FT.ipynb`](./finetuning/Qwen3_5_0.8B_FT.ipynb) | Separate Olive LoRA fine-tuning, CUDA ONNX export, and validation workflow |

The 4B Mobius conversion notebook uses `/content/onnx_outputs` on Colab and
`.mobius_colab_run/onnx_outputs` outside Colab. Generated models, downloaded GGUF
files, virtual environments, archives, and executed notebooks are excluded from Git
because the ONNX external-data file is approximately 2.6 GB.

## Qwen3.5-0.8B fine-tuning with Microsoft Olive

The [fine-tuning notebook](./finetuning/Qwen3_5_0.8B_FT.ipynb) is a **separate
GPU workflow** from the 4B-Hmm GGUF-to-CPU-ONNX conversion described above. It
starts with `Qwen/Qwen3.5-0.8B`, builds Microsoft Olive from pinned Git revision
`2fbeaf4316930d62bf7b85658ccc6e752d4b6f4c`, and follows the
[Olive fine-tune CLI guide](https://microsoft.github.io/Olive/how-to/cli/cli-finetune.html).
Use Linux with an NVIDIA CUDA GPU supporting BF16 (for example, a suitable Colab
runtime); this is not a macOS CPU/MPS training workflow. The notebook pins CUDA
12-compatible ONNX Runtime packages for export. Restart the runtime after installing
or changing native runtime packages.

Run the notebook cells in order when you are ready to install dependencies, download
model/data, train, or export. It loads
[`n4ze3m/typed-decisions-synth`](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth),
preserves the dataset's **case-level** train/validation split, and creates one
option-letter example per question. Choice options are shuffled deterministically;
samples exceeding 768 tokens are discarded. `olive finetune --dry_run` generates a
version-matched configuration, which the notebook changes to `line-by-line` training
with the original validation split. The training cell then calls Olive in the
notebook's Python kernel. It uses BF16 LoRA (rank 32, alpha 64), learning rate
`1e-4`, and one epoch.

Training produces a PEFT adapter under `olive_typed_decisions/finetuned/adapter`.
Subsequent **optional** cells use Olive `ModelBuilder` to export the base model
with that adapter to `olive_typed_decisions/onnx_fp16_cuda` as a CUDA FP16
ONNX Runtime GenAI model, run a sample inference and first-token probability
checks, and optionally compute validation accuracy, NLL, and ECE. Export artifacts
include `model.onnx`, external weights, tokenizer files, and `genai_config.json`.
An optional upload cell creates a model card and prompts for a Hugging Face login;
use an account and token with write access to the target repository. Never place
a token in the notebook or repository.

The question format and option shuffling are inspired by
[Hmm's training script](https://github.com/n4ze3m/hmm/blob/main/training/train.py),
but **the training objectives differ**: Olive applies full-sequence language-model
SFT on the gold answer; it does not train Hmm's candidate-letter-only hard/teacher
soft-label mixture or reshuffle options every step. Teacher probabilities are
retained for analysis only. Olive's full-sequence `eval_loss` is not the
candidate-letter NLL, accuracy, or ECE, and no matching calibration or benchmark
result is claimed here. Training, export, evaluation, and upload must be run and
verified in a suitable GPU environment.

## Reproducibility

The conversion is pinned to:

| Component | Revision |
|---|---|
| GGUF repository | `n4ze3m/Qwen3.5-4B-Hmm` |
| GGUF revision | `c27fa3c627dfaced343c6ba9d3a0d00243a3be51` |
| GGUF filename | `Qwen3.5-4B-Hmm-Q4_K_M.gguf` |
| GGUF SHA-256 | `5e03cb057049c56b421bd3c506d77fd8e0a77996bb8464148a02cfc3caac5229` |
| Mobius revision | `6b27a3f08b8b5d08ba9b14b416e3b435942bb0bd` |
| Tokenizer repository | `Qwen/Qwen3.5-4B` |
| Tokenizer revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

The notebook builds a Mobius wheel from the pinned Git source instead of installing
a prebuilt Mobius package.

## Quick start

Run the notebook in a high-memory Colab CPU runtime:

```text
Qwen3_5_4B_Hmm_GGUF_to_CPU_ONNX_Mobius_fixed.ipynb
```

For the already converted local model:

```bash
python run_hmm_onnx.py \
  --model-dir .mobius_colab_run/onnx_outputs
```

To provide a request from a file:

```bash
python run_hmm_onnx.py \
  --model-dir .mobius_colab_run/onnx_outputs \
  --request request.json
```

Example `request.json`:

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

Python usage:

```python
from pathlib import Path

from run_hmm_onnx import HmmOnnx

model = HmmOnnx(Path(".mobius_colab_run/onnx_outputs"))
response = model.decide(
    {
        "state": "Help! My payouts have failed for 3 days.",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this message convey urgency?",
            }
        },
    }
)
print(response)
```

## Decision types

### `noul`

A nullable-style yes/no probability. The returned `noul` value is the normalized
probability of the `true` option.

```json
{
  "type": "noul",
  "noul": 0.9745
}
```

### `choice`

Returns the highest-probability key, all normalized option probabilities, and a
confidence value measuring how peaked the distribution is.

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": {
    "billing": 0.7216,
    "technical": 0.2759,
    "sales": 0.0025
  },
  "confidence": 0.5823
}
```

### `score`

Returns the expected numeric level, the level legend, normalized probabilities, and
confidence.

```json
{
  "type": "score",
  "score": 2.6653,
  "legend": {
    "0": "Not urgent",
    "1": "Normal",
    "2": "Urgent",
    "3": "Critical"
  },
  "probabilities": {
    "0": 0.0061,
    "1": 0.0361,
    "2": 0.2443,
    "3": 0.7135
  },
  "confidence": 0.618
}
```

Hmm is trained with 26 option letters. The runner accepts up to 255 choice options by
splitting larger sets into groups of 25 and adding a `none_of_these` option to each
group, matching the reference Hmm server strategy.

## Verified result

The generated model was executed directly on an Apple Silicon CPU with ONNX Runtime:

| Check | Result |
|---|---|
| ONNX Runtime | `1.30.0` |
| Execution provider | `CPUExecutionProvider` |
| ONNX checker | Passed |
| Stateful prefill/decode | Passed |
| Hmm typed-decision test | Passed |
| Model graph | 618 KB |
| External model data | 2.6 GB |
| Example input tokens | 244 |
| Three-question inference time | 10.24 seconds |

Observed example output:

- Urgent probability: `0.9745`
- Department: `billing`
- Billing probability: `0.7216`
- Priority score: `2.6653` on a `0..3` scale

Performance depends on CPU, thread count, memory bandwidth, prompt length, and ONNX
Runtime version.

## Important limitations

- Hmm is an experimental fine-tune and its model card advises against using it as the
  only safety gate or for important decisions.
- Hmm reproduces the System One typed-decision idea and request style, but processes
  one question per model pass rather than reproducing Jev's native parallel sampler.
- The output is constrained by adapter code, not by a mathematical type guarantee in
  the underlying Qwen model.
- The probabilities are option-letter probabilities normalized over the supplied
  options. They should be evaluated for calibration on the target workload.
- Hmm was trained in English and primarily on prompts up to 768 tokens.
- Mobius currently lists `qwen35` graph and quantized import as supported while its
  representative real-weight runtime evidence remains pending. This project therefore
  validates the exact converted artifact directly with ONNX Runtime.
- Do not separate `model.onnx` from `model.onnx.data`.

## References and citation

Primary references:

1. TypeSafe AI, [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
2. Muhammed Nazeem, [Hmm (Qwen3.5-4B)](https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm)
3. Microsoft, [ONNXRuntime Mobius](https://github.com/onnxruntime/mobius)
4. Microsoft, [Mobius GGUF import documentation](https://onnxruntime.github.io/mobius/api/build_from_gguf.html)
5. Qwen Team, [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

When using the converted artifact, cite the original Hmm model and acknowledge the
Mobius conversion path. The Hmm model card provides:

```bibtex
@misc{nazeem2026hmm,
  author = {Muhammed Nazeem},
  title  = {Hmm: a small open model for typed decisions},
  year   = {2026},
  url    = {https://github.com/n4ze3m/hmm}
}
```

A suggested citation for this conversion workflow is:

```bibtex
@software{qwen35_hmm_mobius_onnx,
  title  = {Qwen3.5-4B-Hmm GGUF to CPU ONNX with ONNXRuntime Mobius},
  year   = {2026},
  note   = {Conversion and direct CPU inference workflow},
  url    = {https://github.com/onnxruntime/mobius}
}
```

Use the original project or repository URL in the second entry if this workflow is
published.
