# 从 Jev 到你的笔记本：用 Mobius 在 ONNX 中构建「System One」决策模型

*一位技术布道师的实践笔记：认识 ONNXRuntime Mobius、它与 ONNX Runtime GenAI Model Builder 的区别，以及一个把开源「Jev 风格」决策模型转换成 CPU ONNX 的完整示例。*

---

## 0. 为什么是现在写这篇

同一周里有两件事同时撞进了我的视野。

第一件，TypeSafe AI 发布了 **[《Introducing System One Models & Jev》](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**（2026 年 9 月 15 日），开发者社区立刻吵开了。它的定位很不一样：Jev 不是又一个聊天模型，而是被描述为 *"a new class of frontier models built to make fast, structured decisions that software can use directly"*（一类全新的前沿模型，专为做出软件可直接使用的、快速的结构化决策而生）。原文把 Jev 比作 **"a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out"** —— 一次前沿智能的函数调用：输入非结构化状态，输出带类型的概率决策。并且明确说，虽然 *"Jev gives up string generation"*（Jev 放弃了字符串生成），但它 *"is optimized for structured outputs and can't hallucinate"*（为结构化输出而优化，并且不会产生幻觉）。

真正让所有人侧目的是那几个数字：端到端响应时间 **70ms–500ms**，而前沿 LLM 是 3–329 秒；输入 token **$0.042/MTok**，输出 token 免费；每个答案都带有**校准过的概率** —— *"higher confidence means higher accuracy"*（置信度越高，准确率越高）。连名字本身都是一个论断：TypeSafe 说 Jev 取自 **William Stanley Jevons**（杰文斯），因为他们预期 *"every order of magnitude drop in the cost of intelligence unlocks orders of magnitude more use cases"*（智能成本每下降一个数量级，就会解锁数量级更多的使用场景）。

第二件事，对所有在微软 AI 技术栈上做开发的人更重要：**[onnxruntime/mobius](https://github.com/onnxruntime/mobius)** 正在以惊人的速度迭代。正是这块基础设施，让「把这样一个模型——无论它以什么格式发布——变成 ONNX 图，跑在你自己掌控的硬件上」变成一件现实的事。

于是我动手搭了这座桥：**[kinfey/JevONNX](https://github.com/kinfey/JevONNX)**。它拿一个开源社区微调出来的、复刻了 System One *请求形态* 的模型，用 Mobius 把它从 GGUF 转换成 CPU ONNX，然后在本地提供带类型的概率决策服务。

这篇文章讲三件事：Mobius 是什么、它和你可能已经在用的 ONNX Runtime GenAI Model Builder 有什么区别，以及 JevONNX 这个示例是怎么把它们串起来的。

---

## 1. Mobius 是什么？

**Mobius 是「在 ONNX *里面* 构建生成式模型」。** 这个介词就是它全部的思想。

按仓库自己的说法，Mobius 提供的是 *"ONNX model definitions for GenAI using the onnxscript.nn API"* —— 覆盖 *"LLMs, MoE, multimodal, encoder-only, encoder-decoder, vision, audio, and diffusion models"*，全部 *"built directly as ONNX graphs using `onnxscript.nn.Module`"*（直接用 `onnxscript.nn.Module` 构建为 ONNX 图）。

其中最关键、值得读两遍的一句话是：

> *"Rather than tracing or exporting PyTorch models, it constructs the ONNX graph declaratively, then applies pretrained HuggingFace weights."*
>
> （它不是去 trace 或导出 PyTorch 模型，而是以声明式的方式构建 ONNX 图，然后再把预训练的 HuggingFace 权重应用上去。）

### 为什么「声明式构建」是件大事

任何用传统方式导出过 ONNX 的人都知道那些坑。你跑 `torch.onnx.export`，tracer 只会沿着 Python 控制流执行**一条**具体路径；任何动态的东西——KV cache 的分支、用 Python `if` 算出来的 rotary embedding、滑动窗口 mask——要么被悄悄固化成常量，要么炸成一坨没法读的子图。然后你就得花一周时间 diff 计算图。

Mobius 把这件事反过来了。**图本身是真相来源**，用可组合的模块写出来，权重是之后才贴上去的。README 里的架构是干净的四层：

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

- **Components（组件）** —— `onnxscript.nn.Module` 构建块（Attention、MLP、DecoderLayer、RoPE、VisionEncoder、MoELayer……）
- **Models（模型）** —— 由组件组合而成的完整架构
- **Tasks（任务）** —— 定义 ONNX 图的 I/O 契约（输入、输出、KV cache）
- **Registry（注册表）** —— 把 HuggingFace 的 `model_type` 字符串映射到模型类

于是「支持一个新架构」从*调试导出器*变成了*写一份模型定义*。Mobius 甚至在 `.agents/skills/` 里内置了开发者 skills（`adding-a-new-model`、`reusable-components`、`moe-models`、`multimodal-models`、`writing-tests`、`writing-rewrite-rules`），让 AI 编码代理可以跟你一起干这个活。

### 覆盖面

这不是玩具。仓库声称支持 **290+ 种 Transformers 模型类型、10 种 Diffusers 组件类型，横跨 40+ 种任务类型和 100+ 个可复用组件**：

| 类别 | 示例 |
|---|---|
| 文本生成 | Llama 2/3/4、Mistral、Qwen 2/2.5/3/3.5/3.6、Phi-3/3.5、Gemma 1/2/3/4、Granite、GPT-2、OPT、OLMo、SmolLM3 |
| 混合专家 MoE | PhiMoE、GPTOSS、Mixtral、OLMoE、DeepSeek-V2/V3、Qwen2-MoE、Qwen3-MoE、Qwen3-Next、GLM-4-MoE、Arctic、DBRX、Jamba |
| 多模态 | Gemma 3/4、Phi-4MM（视觉 + 音频 + LoRA）、LLaVA、InternVL2、MiniCPM-V 4.6、Qwen2.5-VL、Qwen3-VL、Pixtral |
| 纯编码器 | BERT、RoBERTa、ALBERT、DeBERTa、DistilBERT、ELECTRA、XLNet |
| 编码器-解码器 | BART、T5/mT5、Marian、M2M-100、Pegasus、BigBird-Pegasus |
| 语音转文本 | Whisper、Moonshine、Moonshine Streaming、FastConformer-RNNT、FunASR、GLM-ASR、Qwen3-ASR、SenseVoice |
| 音频 | Wav2Vec2、HuBERT、WavLM、SpeechT5 |
| 视觉 | ViT、BEiT、DeiT、DINOv2、Swin、CLIP、SigLIP |
| 扩散模型 | Stable Diffusion（UNet + VAE + ControlNet）、Flux、SD3、DiT、QwenImage / Qwen-Image-Edit-2509、HunyuanDiT、CogVideoX |
| 适配器 | T2I-Adapter、IP-Adapter |

### 五分钟上手

```bash
pip install -e .
```

```python
from mobius import build

pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama-3.2-1b/")
```

当你事先知道最大序列长度时，可以开启静态 KV cache：

```python
from mobius import build, CausalLMTask

task = CausalLMTask(static_cache=True, max_seq_len=2048)
pkg = build("meta-llama/Llama-3.2-1B", task=task)
pkg.save("output/llama-3.2-1b-static/")
```

EP 感知优化 —— 图本身就是针对某个执行提供程序调优的，*"each with the right set of fused kernels and lowering passes applied automatically"*（每种 EP 都会自动应用对应的融合内核与下降 pass）：

```python
# CUDA：GQA 融合、SkipLayerNorm、PackQKV
pkg = build("meta-llama/Llama-3.2-1B", execution_provider="cuda", dtype="f16")

# WebGPU：GQA 融合，Shape 算子被替换为可移植的替代实现
pkg = build("meta-llama/Llama-3.2-1B", execution_provider="webgpu", dtype="f16")
```

还有 CLI，这也是后面示例要用的：

```bash
mobius build --model Qwen/Qwen2.5-0.5B --output output_dir/
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ --ep cuda --dtype f16
mobius build --model openai/whisper-tiny --output output_dir/   # 产出 encoder/ + decoder/
```

构建模式开关采用 cargo 风格的 `--features`，可用特性为 `static-cache`、`fp8-kv-cache`、`prune-prefill-prefix` 和 `text-only`：

```bash
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ \
  --features static-cache,prune-prefill-prefix --max-seq-len 2048
```

`--release` 会剥离构建期的调试与溯源元数据以减小模型体积，同时保留以 `mobius.` 为前缀的功能性元数据。

---

## 2. Mobius 与 ONNXRuntime-genai Model Builder 的区别

每次我演示 Mobius，这都是第一个被问到的问题，所以我们说清楚。**两者都是微软的工具，都产出 ONNX，但它们不是竞争关系——它们站在同一条流水线的不同位置上。**

**[ONNX Runtime GenAI Model Builder](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models)** 对自己的定位是：*"for quickly creating optimized and quantized ONNX models within a few minutes that run with ONNX Runtime GenAI"*（在几分钟内快速创建可在 ONNX Runtime GenAI 上运行的、已优化与量化的 ONNX 模型）。它的产出是一个**可直接部署的包**：`model.onnx` + `genai_config.json` + tokenizer 文件，专为 `onnxruntime-genai` 运行时接好线。它支持的架构清单是明确策展过的——AMD OLMo、ChatGLM、DeepSeek、ERNIE 4.5、Gemma、gpt-oss、Granite、Granite MoE Hybrid、HunYuan Dense V1、InternLM2、LFM2、LFM2 MoE、Llama、Mistral、Nemotron、Phi、Qwen、SmolLM3、Whisper——README 里也直说了：*"It is intended for supporting the latest, popular state-of-the-art models."*（它的目标是支持最新的、流行的 SOTA 模型。）

而 Mobius 是一个**模型定义库**：用一套组件代数来把生成式架构描述成 ONNX 图，再通过注册表把 HuggingFace 的 `model_type` 映射到模型类。

我会这样区分它们：

| 维度 | ORT GenAI Model Builder | Mobius |
|---|---|---|
| **首要目标** | 快速产出可部署的 GenAI 模型包 | 定义并构建 ONNX 图本身 |
| **产出契约** | `model.onnx` + `genai_config.json` + tokenizer，面向 ORT GenAI 运行时 | ONNX 模型包；运行时由你选（包括普通的 ORT session） |
| **图是怎么来的** | 每个架构一个 builder 子类（`LlamaModel`、`QwenModel`、`Phi3MiniModel`、`WhisperModel`…）重写 `make_layer` / `make_attention` / `make_moe` | 声明式的 `onnxscript.nn.Module` 组合：Components → Models → Tasks |
| **广度** | 策展的约 19 个热门家族，文本 + Whisper | 290+ Transformers 类型、10 种 Diffusers 组件、40+ 任务，含视觉、音频、扩散 |
| **如何扩展** | 在 `onnxruntime-genai` 代码树里加一个 builder 子类 | 用已有组件拼出一份模型定义；仓库内置 `.agents/skills/` 指引 |
| **调优面** | 丰富的 `--extra_options`，覆盖量化与运行时行为（QMoE、paged attention、KV cache 量化、QDQ、CUDA/WebGPU graph capture…） | `--features` 构建开关 + `execution_provider=` EP 感知图下降 + `dtype` 控制 |
| **扩散 / 多组件管线** | 不是它的目标 | 一等公民（`--model Qwen/Qwen-Image-2512` 会构建全部组件） |
| **我什么时候选它** | 「我手上是 Phi/Llama/Qwen 聊天模型，今天就要一个跑在 ORT GenAI 上的 int4 CPU/CUDA 包」 | 「我的架构是混合的、非常规的、多模态或扩散的，或者我需要对导出内容做图级别的控制」 |

值得注意的是，两者在能力上确实有重叠。Model Builder 同样能吃 GGUF，也支持 Qwen3.5 级别的特性（compact state updates、recurrent 算子选择）。Mobius 同样能产出 ORT-GenAI 兼容的元数据——不过仓库对边界很诚实：对于 Mage-VL，*"ORT GenAI export is currently rejected because the runtime cannot supply its required `patch_positions` input or Mage-VL's 1D decoder positions."*（ORT GenAI 导出目前被拒绝，因为该运行时无法提供它所需的 `patch_positions` 输入或 Mage-VL 的 1D decoder positions。）

这一句话就把分界线画得清清楚楚：**Model Builder 的上限是「GenAI 运行时能表达什么」；Mobius 的上限是「ONNX 能表达什么」。**

**作为布道师，我的经验法则是：** 如果你的模型在 Model Builder 的支持列表上，而且你要的就是 GenAI 运行时那套生成循环——用 Model Builder，它更快，而且直接给你一个完整的包。一旦你的模型不在那个列表上，或者你需要把「图」本身当成可推理、可改写的一等产物——那就该 Mobius 出场了。

JevONNX 这个示例，正好属于第二类。我们来看为什么。

---

## 3. 示例：把一个 Jev 风格的决策模型转成 CPU ONNX

仓库：**[github.com/kinfey/JevONNX](https://github.com/kinfey/JevONNX)**

### 3.1 先把话说清楚

Jev 是 TypeSafe 的专有模型。你下载不到它，也就无从转换，这个项目也不打算假装可以。

社区做的事情反而更有意思：`n4ze3m/Qwen3.5-4B-Hmm` 是 **"an experimental open-model attempt to reproduce the central idea and request shape of the System One workflow introduced with Jev"**（一次用开源模型复刻 Jev 所引入的 System One 工作流之核心思想与请求形态的实验）。它的做法是 *"fine-tuning Qwen3.5-4B to answer typed questions through option probabilities instead of ordinary generated text"*（微调 Qwen3.5-4B，使其用选项概率而非普通生成文本来回答带类型的问题）。

JevONNX 的 README 对边界写得非常克制而明确，我想把它放在正文里而不是藏在脚注：

> *"Hmm is not Jev, is not affiliated with TypeSafe AI, and does not reproduce Jev's proprietary model architecture, parallel sampler, performance, calibration, or type-safety guarantees. The Hmm model card explicitly states that it is less capable than Jev."*
>
> （Hmm 不是 Jev，与 TypeSafe AI 无任何关联，也没有复刻 Jev 的专有模型架构、并行采样器、性能、校准性或类型安全保证。Hmm 的模型卡明确写明它的能力弱于 Jev。）

以及：*"The reproduction is behavioral rather than architectural."*（这次复刻是行为层面的，而非架构层面的。）

项目对照表里的区别：

| 属性 | Jev | 本项目中的 Hmm |
|---|---|---|
| 模型设计 | 原生 System One Model | 微调 Qwen3.5-4B 以复刻带类型决策的概念 |
| 输出机制 | 原生带类型概率输出 | 首个生成 token 的选项字母概率 |
| 多问题处理 | 为并行采样而设计 | 类 Jev 的请求形态，实现上是每个问题一次模型前向 |
| 输出类型 | 受 schema 约束的结构化值 | `noul`、`choice`、`score` 适配器 |
| 此处运行时 | TypeSafe 服务 | 本地 ONNX Runtime CPU |
| 保证 | 以 TypeSafe 文档为准 | 不声称任何等价保证 |

机制本身清爽得令人愉快：适配器把每个带类型的问题转成一个精确的多选题 prompt，然后 *"reads the first generated-token logits for A, B, C, and subsequent option letters and normalizes them into probabilities"*（读取首个生成 token 在 A、B、C 及后续选项字母上的 logits，并归一化为概率）。没有 JSON 解析、没有「输出格式不对就重试」、没有正则。**那个概率分布本身就是答案。**

### 3.2 为什么这件事非 Mobius 不可

Qwen3.5 是**混合（hybrid）**架构——这正是这个故事必须由 Mobius 来讲的原因。

导出的模型有 **32 个混合解码器层**。线性注意力层携带 `conv_state` 和 `recurrent_state`；全注意力层携带 key/value 缓存。在这个产物里，全注意力层位于索引 **3、7、11、15、19、23、27、31**。

实际后果，引自 README：

> *"The prompt is processed one token at a time while every `present.*` output is fed back into its corresponding `past_key_values.*` input. This differs from a conventional decoder-only example that can prefill an entire prompt in one call."*
>
> （prompt 是逐 token 处理的，每个 `present.*` 输出都要回灌到对应的 `past_key_values.*` 输入。这与常规的 decoder-only 示例不同——后者可以一次调用就把整个 prompt 预填充完。）

这是一个**图形状**的问题。你没法靠一个配置开关糊弄过去：你需要导出图的状态契约是显式的、可检查的——而这恰恰就是声明式构建给你的东西。该实现 *"follows the structure of Mobius `examples/text_generation.py` while adding the hybrid-state handling required by Qwen3.5"*（沿用 Mobius `examples/text_generation.py` 的结构，并加上 Qwen3.5 所需的混合状态处理）。

### 3.3 转换流水线

```
n4ze3m/Qwen3.5-4B-Hmm-Q4_K_M.gguf
        │  SHA-256 校验
        ▼
Mobius GGUF 读取器 / 导入器
        │  元数据与张量映射
        │  Qwen3.5 混合图构建
        │  Q4_K_M → 打包量化目标存储
        │  CPU 执行提供程序优化
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
        │  逐 token 的混合状态预填充
        │  首 token 的 A/B/C/… 概率
        ▼
带类型的决策 JSON 响应
```

转换就是一条命令：

```bash
mobius build-gguf Qwen3.5-4B-Hmm-Q4_K_M.gguf \
  --output onnx_outputs \
  --ep cpu \
  --dtype f32 \
  --release
```

有两个细节值得记住：

- **刻意不使用 `--dequantize`。** 保留量化存储可以 *"substantially reduc[e] the output size compared with a fully dequantized FP32 model"*（相比完全反量化的 FP32 模型显著减小输出体积）。即便如此，外部数据文件仍约 **2.6 GB**——你可以想象反量化后是什么规模。
- **别信预设，去读报告。** *"Q4_K conversion can involve a lossy affine repack; inspect `quantization_report.json` rather than assuming source-preset fidelity."*（Q4_K 转换可能涉及有损的仿射重打包；请检查 `quantization_report.json`，而不要假定与源预设完全保真。）就是这类细节，把「演示」和「上线」区分开来。

### 3.4 可复现性——这部分我真心希望你抄走

README 里把每一个 pin 都列成了表：

| 组件 | 版本 |
|---|---|
| GGUF 仓库 | `n4ze3m/Qwen3.5-4B-Hmm` |
| GGUF revision | `c27fa3c627dfaced343c6ba9d3a0d00243a3be51` |
| GGUF 文件名 | `Qwen3.5-4B-Hmm-Q4_K_M.gguf` |
| GGUF SHA-256 | `5e03cb057049c56b421bd3c506d77fd8e0a77996bb8464148a02cfc3caac5229` |
| Mobius revision | `6b27a3f08b8b5d08ba9b14b416e3b435942bb0bd` |
| Tokenizer 仓库 | `Qwen/Qwen3.5-4B` |
| Tokenizer revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

Notebook 是**从固定的 Git 源码构建 Mobius wheel**，而不是装一个预编译包。对于一个已有 500+ 次提交、仍在高速迭代的项目来说，这不是偏执——这是你下个月还能复现这次转换的唯一办法。

### 3.5 跑起来

项目由两个文件承载：`Qwen3_5_4B_Hmm_GGUF_to_CPU_ONNX_Mobius_fixed.ipynb`（Colab 上的转换与验证流程）和 `run_hmm_onnx.py`（独立的带类型决策运行器）。在高内存的 Colab CPU 运行时里跑 notebook；它在 Colab 上写入 `/content/onnx_outputs`，在 Colab 之外写入 `.mobius_colab_run/onnx_outputs`。

然后：

```bash
python run_hmm_onnx.py --model-dir .mobius_colab_run/onnx_outputs
python run_hmm_onnx.py --model-dir .mobius_colab_run/onnx_outputs --request request.json
```

一个请求 = 状态 + 若干带类型的问题：

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

Python 调用：

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

### 3.6 三种决策类型

**`noul`** —— 可空风格的是/否概率，即 `true` 选项的概率：

```json
{ "type": "noul", "noul": 0.9745 }
```

**`choice`** —— 返回概率最高的 key、所有归一化后的选项概率，以及一个衡量分布陡峭程度的置信度：

```json
{
  "type": "choice",
  "choice": "billing",
  "probabilities": { "billing": 0.7216, "technical": 0.2759, "sales": 0.0025 },
  "confidence": 0.5823
}
```

**`score`** —— 返回期望的数值等级、等级图例、归一化概率和置信度：

```json
{
  "type": "score",
  "score": 2.6653,
  "legend": { "0": "Not urgent", "1": "Normal", "2": "Urgent", "3": "Critical" },
  "probabilities": { "0": 0.0061, "1": 0.0361, "2": 0.2443, "3": 0.7135 },
  "confidence": 0.618
}
```

多看两眼那个 `score`。你拿到的不是一句「这很紧急」，而是 `2.6653`——一个明显偏向 *Critical*、但在 *Urgent* 上仍有实打实权重的分布。这是一个你的路由代码可以设阈值、可以打日志、可以做 A/B 的数字。这就是「一个嘴上说自己很确定的 LLM」和「一个你真的能拿去做校准的分布」之间的差别。

一个约束：**Hmm 是用 26 个选项字母训练的。** 更大的选项集合的处理方式是拆成每组 25 个，并给每组加一个 `none_of_these` 选项——与参考 Hmm server 的策略一致。

### 3.7 它真的跑得起来吗？

跑得起来。README 记录了生成的模型直接在 **Apple Silicon CPU 上用 ONNX Runtime 执行**。一个 4B 的混合架构模型，从 GGUF 转换而来，在笔记本 CPU 上跑带类型决策，不依赖任何服务。

---

## 4. 如果面对一屋子开发者，我会说什么

**即使模型不可迁移，System One 的思想是可迁移的。** 你没法在本地跑 Jev，但那个*形态*——输入状态，输出带概率的类型化决策——今天就能用开源模型加一张 ONNX 图原型出来。用 TypeSafe 的话说，结构化输出 *"slot into ordinary software as fuzzy decision rules: classify, route, score, extract, or branch where hand-written logic is too brittle"*（作为模糊决策规则嵌进普通软件里：在手写逻辑太脆弱的地方做分类、路由、打分、抽取或分支）。这是一种模式，不是一件商品。

**概率才是真正的产品。** 这类模型重要的原因不只是快。TypeSafe 那句话很扎心：*"If a model can do a task 95% of the time but doesn't say when it's in the 5%, it can't automate that task."*（如果一个模型 95% 的情况下能做对某件事，却不告诉你什么时候落在那 5% 里，它就无法自动化这件事。）每一个上线过 LLM 功能的工程师，都撞过这堵墙。

**诚实地设定预期。** JevONNX 是一次教学性质的复刻。它给你的是在本地 CPU 上跑的 `noul`/`choice`/`score`；它**没有**给你 Jev 的校准性、并行采样器或类型安全保证——项目自己的 README 就是这么写的。做 demo 的时候请把这条免责声明一起讲出去。

**Mobius 是值得押注、值得学的基础设施。** 当一个新架构出现时——混合注意力、新颖的 MoE 路由、三模型多模态导出——你要问的问题不再是「导出器支持吗？」，而是「我该组合哪些组件？」。后者是好得多的问题。

---

## 链接

- Mobius —— https://github.com/onnxruntime/mobius （文档：https://onnxruntime.github.io/mobius/）
- ONNX Runtime GenAI Model Builder —— https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models
- JevONNX 示例 —— https://github.com/kinfey/JevONNX
- Introducing System One Models & Jev —— https://typesafe.ai/blog/introducing-system-one-models-and-jev

*Mobius 采用 MIT 许可证。JevONNX 是一个独立实验项目，与 TypeSafe AI 无关联，亦未获其背书。*
