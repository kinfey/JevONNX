# From Chat to Decisions: Bringing SLMs into Application Logic with Microsoft Olive and ONNX

> Starting with a clarification in an automated ordering flow, we explore how Jev-style structured decisions can move closer to the edge—and how fine-tuning Qwen3.5-0.8B provides a bridge from a general experiment to vertical applications.

## 1. Why Jev Matters: Accelerating the Next Action, Not Just the Next Reply

Let us return to the automated ordering scenario we have been exploring.

A customer says, “I’ll have a combo, but make the drink not too sweet.”

A conversational model can produce a friendly response. For the ordering application, however, the real work is just beginning. Has the customer selected a particular combo? Does “not too sweet” map to a supported customization? Should the application ask a question, update the cart, or move to confirmation?

**The application does not need another paragraph. It needs a decision that advances the workflow.**

Traditional rules are excellent at explicit constraints: unavailable items cannot be ordered, checkout requires confirmation, and payment requires authorization. Models should not replace these rules. But as inputs become ambiguous and context-dependent, rule trees accumulate branches. At the other extreme, sending every small routing decision through a large model’s reasoning and generation loop can add unnecessary waiting.

This is how I think about the value of Jev: **turn semantic understanding into structured behavioral decisions, shortening the distance between understanding and action.**

An ordering application might decompose the conversation into a few bounded questions:

| Decision | Output type | Effect on the workflow |
|---|---|---|
| Is a critical preference still ambiguous? | `noul`: probability for a Boolean question | Decide whether to clarify |
| Which stage should handle the request next? | `choice`: candidate action and distribution | Route to clarification, cart editing, or confirmation |
| How much human intervention is needed? | `score`: expected value over ordered levels | Decide whether to escalate |

These outputs are not permissions. A model can recommend confirmation, but that recommendation must not bypass inventory checks, allergen rules, or payment authorization. The application still needs explicit state transitions, constraints, and tool boundaries.

The lesson I take from our ordering exploration is that **structure does more than make an answer easier to parse: it can remove unnecessary decision steps from an agent’s workflow.** If questions, options, and execution boundaries are defined in advance, the model does not have to reinvent the process on every turn.

One distinction matters before we continue. This project does not fine-tune an official Jev model or reproduce its private implementation. It borrows the Jev-style typed-decision interface idea, follows the prompt conventions of the open-source [Hmm project](https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm), and explores a deployable component built from our own small model.

## 2. Where Edge SLMs Fit: Put the Decision Close to the Action

Beyond basic rules, the opportunity that interests me most is in vertical applications: can a model make a useful semantic decision where the action actually happens?

An SLM—a small language model—is not automatically an edge-ready model. Model size, context length, runtime, memory, and hardware all determine whether it fits a target device. Nevertheless, small models suggest a useful division of responsibilities:

**Use large models for open-ended problems, small models for frequent and bounded semantic decisions, and deterministic code for constraints and execution.**

### From ordering terminals to embodied systems

At an ordering terminal, a local model could recognize a modification request, assess whether required information is complete, and select the appropriate business module. Some decisions could continue during network disruption, and sensitive context could avoid unnecessary transmission. Those benefits depend on the application’s complete data flow, not merely on storing model weights locally.

In an embodied system, similar questions become task-level intent decisions: “Is the user asking the robot to approach, or only describing a location?” “Should the task continue, or wait for clarification?”

There is an essential boundary here. A language model may participate in task-level decisions, but it should not be the sole mechanism for collision avoidance, emergency stopping, or low-level motion control.

Millisecond-scale responsiveness is a worthwhile engineering objective—not an automatic consequence of using a small model or exporting to ONNX. We need to measure the complete path on the target device: input processing, inference, policy evaluation, and tool execution.

### From one assistant to multi-agent orchestration

Multi-agent systems offer another opportunity. A complex request may involve research, planning, execution, and verification, but not every handoff requires an open-ended discussion.

A local decision component might determine:

- Which agent should receive the request?
- Is the information required for execution complete?
- Does a tool result need independent review?
- Should the workflow continue, retry, or ask a human?

I am exploring how hosted agents can handle longer orchestration flows while constrained autonomous execution moves work forward. Small models could serve as frequent, local decision points within that design. “Autonomous” does not mean unrestricted: tool permissions, timeouts, retry budgets, and human approval remain application responsibilities.

This is a compelling role for ONNX: package a learned decision capability into an artifact that an inference runtime can load, without requiring every deployment target to carry the full training stack.

However, **ONNX is a deployment path, not a guarantee of portability or performance across all hardware.** This workflow exports a CUDA FP16 model. Moving it to a CPU, mobile device, or NPU requires compatible execution providers and operators, and potentially re-export, quantization, and regression evaluation.

## 3. Fine-Tune a General Decision Component, Then Specialize It

Rather than beginning with a complex industry system, I started with a general typed-decision experiment:

**Qwen3.5-0.8B → Olive LoRA fine-tuning → Olive ModelBuilder export → ONNX probability testing → a Hugging Face publishing workflow.**

The implementation lives in [Qwen3_5_0.8B_FT.ipynb](../Qwen3_5_0.8B_FT.ipynb). Its purpose is not to claim that a notebook solves a vertical market. It makes the path from data to a deployable decision capability concrete.

### Stop one: learn a bounded choice, not a long answer

The base model is [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), and the dataset is [n4ze3m/typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth).

A state can contain several questions. Preprocessing expands each question into a training example while preserving the dataset’s case-level training and validation split. That prevents questions about the same state from leaking across the boundary.

Each question becomes a choice among option letters:

```text
State (data to evaluate):
Help! My payouts have failed for 3 days. I need the money today.

Question:
Which team should handle this?

Options:
A: billing — Payments, invoicing, refunds
B: technical — Bugs, outages, integrations
C: sales — Pricing, upgrades, new accounts
Return only the option letter.
```

`choice` represents categorical options, `noul` uses `false / true`, and `score` uses ordered levels. During preprocessing, choice options are shuffled deterministically using the state and question ID to reduce fixed-position bias. Score levels and Boolean options keep their intended ordering.

Several small details protect the contract: use the same chat template, disable thinking, verify that training text begins with the inference prompt, verify that each option letter is one token, and discard examples longer than 768 tokens rather than blindly truncating them.

This is not cosmetic data cleanup. It defines the protocol between the model and its application.

### Stop two: turn experiment settings into an inspectable Olive workflow

The notebook builds Microsoft Olive from a fixed source revision. It uses `olive finetune --dry_run` to generate a version-matched configuration, then adjusts preprocessing and validation.

| Setting | Notebook configuration |
|---|---|
| Fine-tuning method | LoRA, rank 32, alpha 64 |
| Target modules | `q_proj/k_proj/v_proj/o_proj` and `gate_proj/up_proj/down_proj` |
| Learning rate and epochs | `1e-4`, one epoch |
| Sequence length | 768 |
| Per-device batch / gradient accumulation | 1 / 16 |
| Training / export precision | BF16 / FP16 |
| Preprocessing | `line-by-line`, one sequence per question |

Replacing the default JOIN strategy avoids concatenating independent questions into continuous text. Before training, `RunConfig.model_validate(...)` checks the configuration. Execution uses Olive’s Python API so failures remain visible in the notebook rather than being hidden behind a generic `CalledProcessError`.

**An important difference must remain explicit: this Olive workflow uses standard full-sequence language-model SFT, not Hmm’s candidate-letter hard-label / teacher soft-label mixture.** Teacher probabilities are retained for analysis but do not contribute to this training loss. Consequently, Hmm’s accuracy and calibration results cannot be claimed for this 0.8B model.

### Stop three: export the runtime contract, not just the graph

A LoRA adapter is not the deployment endpoint. The base model, adapter, tokenizer, configuration, and runtime must agree.

The notebook uses Olive `ModelBuilder` to export the base model and adapter into CUDA FP16 ONNX. The result includes more than an `.onnx` file: external weights, tokenizer resources, and `genai_config.json` are part of the deployable unit. Consumers should load the complete directory.

The most instructive part of the process was navigating version boundaries:

| Boundary encountered | Approach in this workflow |
|---|---|
| An unnecessary old TorchAO package conflicted with Transformers | Remove it after dependency installation; standard BF16 LoRA does not require it |
| Qwen3.5 kernels fell back to reference implementations | Install and check causal-conv1d and Flash Linear Attention |
| Export requested authentication for a public model | Explicitly enable anonymous model access |
| Qwen3.5 stored EOS information in `text_config` | Add the top-level value to the local export configuration |
| Newer CUDA packages did not match the runtime environment | Pin the notebook’s ORT GPU 1.26.0 / GenAI CUDA 0.14.0 combination |
| The Olive revision expected newer GenAI Builder APIs | Add version-specific compatibility handling for this FP16 export path |
| Text export behaved like a multimodal decoder | Set `exclude_embeds=False` to retain token embeddings |

These are records of this environment, not universal patches. Compatibility code is technical debt: when upgrading, revisit the interfaces and outputs rather than copying the workaround indefinitely.

Pinning an Olive commit also does not make the complete experiment reproducible. A stronger release record should include base-model and dataset revisions, the full dependency set, GPU and driver details, and the final publishing commit.

### Stop four: ask two different questions in testing

The first question is: **does the model run?**

The standalone [Python test script](../test_qwen35_onnx.py) loads the export, reads a validation example, generates an answer, and extracts an option letter. This is a smoke test for the model, tokenizer, and generation loop—not the end of quality evaluation.

The second question is: **can the application use the decision?**

Following Hmm’s approach, the ONNX probability cell reads logits at the first generated position after the prompt, selects only the valid option-letter logits, and applies softmax:

```python
# candidate_logits contains only valid first-token option logits.
shifted = candidate_logits - candidate_logits.max()
probabilities = np.exp(shifted)
probabilities /= probabilities.sum()
```

For options A, B, and C, this is a distribution over those three choices. It does not establish that the candidate set is complete or provide absolute confidence across every possible answer.

The application then interprets the result: `noul` reads `P(true)`, `choice` selects the highest-probability option, and `score` computes an expected value over ordered levels. The example’s confidence formula, `(n × p_max - 1) / (n - 1)`, measures how concentrated the distribution is. **It is not a calibrated probability of correctness.**

Evaluation therefore needs several lenses. Accuracy, candidate NLL, and ECE describe decision quality. P50/P95 latency, peak memory, fallback rate, and task completion on the target device describe application value.

The notebook’s full validation-metric cell currently uses the PyTorch/PEFT path; the ONNX cell tests example probabilities. The former is not an evaluation report for the latter. Comparing candidate distributions and actual decisions before and after export remains necessary.

This article does not claim unrecorded training times, accuracy improvements, or demonstrated millisecond edge latency.

### Stop five: publication begins the vertical journey

The notebook contains a workflow to upload the ONNX directory and a Model Card to [`lokinfey/Qwen3_5_0.8B_jev`](https://huggingface.co/lokinfey/Qwen3_5_0.8B_jev). This is the configured publishing destination, not a claim that this article verified the remote upload.

The Model Card should explain the base model, dataset, training-objective differences, runtime versions, intended use, and outstanding evaluation. Authentication uses interactive token entry rather than embedding credentials in source files or commit history.

Once the general pipeline exists, specialization becomes the important work. In ordering, that means defining real business states and actions: menu version, cart contents, modification history, missing information, mandatory clarification, and operations that always require customer confirmation.

Vertical data should reflect those boundaries, with validation splits based on conversations or business entities. Low confidence, out-of-distribution inputs, and rule conflicts need explicit rejection or escalation paths. The objective is not to force a choice for every request. It is to help the system move forward appropriately within known limits.

## 4. Conclusion: The Opportunity Is a Shorter Decision Path, Not a Smaller Chat Window

An ordering clarification, a robot’s task-level intent check, and a local handoff in a multi-agent workflow share the same underlying problem: the system has tools, but it still needs to decide when to use them, who should act, and when to stop.

Jev-style structured decisions offer an interface perspective. SLMs offer the possibility of moving some semantic judgments closer to the device. Microsoft Olive organizes fine-tuning and export into a workflow. ONNX Runtime provides an execution path for deployment.

Together, they suggest something more useful than a smaller chatbot: **a measurable, constrained, replaceable decision component.**

The direction I want to keep testing combines hosted agents for long-running coordination, edge small models for frequent local decisions, and deterministic code for business and safety boundaries. Success should be measured less by whether the reply sounds intelligent and more by whether the system completes the right action faster and more reliably.

That is the next step from conversation to decisions.

---

**Implementation and references**

- [Fine-tuning notebook](../Qwen3_5_0.8B_FT.ipynb)
- [Microsoft Olive](https://github.com/microsoft/Olive)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)
- [Qwen3.5-0.8B base model](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Hmm model card and prompt format](https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm)
- [typed-decisions-synth dataset](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth)
