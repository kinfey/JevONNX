# Qwen3.5-4B-Hmm GGUF 转 CPU ONNX

[English](./README.md)

本项目使用
[Microsoft ONNXRuntime Mobius](https://github.com/onnxruntime/mobius)，将
[`n4ze3m/Qwen3.5-4B-Hmm`](https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm)
从 GGUF 转换为面向 CPU 的 ONNX 模型，并提供其类型化概率决策接口。Hmm 是一个
开放模型实验，尝试复现 TypeSafe AI 在 Jev 中提出的 System One 核心思想和请求
形式，相关背景见
[《Introducing System One Models & Jev》](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
。

Hmm 通过微调 Qwen3.5-4B，让模型以选项概率而不是普通生成文本回答类型化问题，
从而复现这一产品思想。但它仍然是独立实验：Hmm 不是 Jev，与 TypeSafe AI
没有关联，也没有复现 Jev 的专有模型架构、并行采样器、性能、校准或类型安全
保证。Hmm 模型卡也明确说明其能力不及 Jev。

## 项目内容

- 可在 Colab 中运行、从源代码构建 Mobius 的 notebook。
- 下载并使用 SHA-256 校验指定的 GGUF 文件。
- 面向 ONNX Runtime CPU 的 GGUF → ONNX 转换。
- 在 Mobius 支持的情况下保留量化权重存储。
- ONNX 图验证与 `CPUExecutionProvider` 直接执行。
- 支持 Qwen3.5 DeltaNet 与完整注意力混合架构的有状态推理。
- 与 Hmm 的 `noul`、`choice` 和 `score` 结构兼容的 Python API。
- 可直接运行转换后 ONNX 模型的独立脚本。

## System One、Jev 与 Hmm

TypeSafe 将 System One Model 描述为一种面向快速结构化决策的模型：输入非结构化
或结构化状态，输出软件可直接使用的类型化结果、概率和置信度。Jev 是 TypeSafe
原生的 System One Model，其目标是并行产生结构化结果，而不是生成自由文本。

Hmm 是一个经过微调的 Qwen3.5-4B 实验，目标是在开放模型中复现 Jev 的核心
System One 思想：提供状态、提出类型化问题，并得到带概率的决策而不是自然语言
长文本。其本地 server 特意遵循与 Jev 类似的请求形式，便于开发者试验这一概念。

这种复现是行为和接口层面的，而不是模型架构层面的。Hmm 将每个问题转换为严格
的多选提示，然后读取第一个生成 token 对 `A`、`B`、`C` 等选项字母的 logits，
并将它们归一化为概率。

两者的区别如下：

| 属性 | Jev | 本项目中的 Hmm |
|---|---|---|
| 模型设计 | 原生 System One Model | 为复现类型化决策思想而微调的 Qwen3.5-4B |
| 输出机制 | 原生类型化概率输出 | 首 token 的选项字母概率 |
| 多问题处理 | 面向并行采样设计 | Jev 风格请求形式，每个问题执行一次模型 |
| 输出类型 | 由 schema 约束的结构化值 | `noul`、`choice`、`score` 适配层 |
| 本项目运行时 | TypeSafe 服务 | 本地 ONNX Runtime CPU |
| 保证 | 以 TypeSafe 文档为准 | 不声明具有等价保证 |

## 转换架构

```text
n4ze3m/Qwen3.5-4B-Hmm-Q4_K_M.gguf
                  |
                  | SHA-256 校验
                  v
        Mobius GGUF reader/importer
                  |
                  | 元数据与张量映射
                  | 构建 Qwen3.5 混合架构计算图
                  | Q4_K_M -> packed 量化目标存储
                  | CPU execution-provider 优化
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
                  | 逐 token 混合状态 prefill
                  | 第一个 token 的 A/B/C/... 概率
                  v
           类型化决策 JSON
```

转换命令：

```bash
mobius build-gguf Qwen3.5-4B-Hmm-Q4_K_M.gguf \
  --output onnx_outputs \
  --ep cpu \
  --dtype f32 \
  --release
```

这里有意不使用 `--dequantize`。Mobius 会尽可能使用 packed 量化存储，相比完整
FP32 模型可显著减小输出体积。Q4_K 转换可能包含有损 affine repack，应该查看
`quantization_report.json`，而不是假设转换后与源量化预设完全一致。

## Qwen3.5 运行时状态

导出的模型包含 32 个混合解码层：

- 线性注意力层维护 `conv_state` 和 `recurrent_state`。
- 完整注意力层维护 key/value cache。
- 在该模型中，完整注意力层索引为 3、7、11、15、19、23、27 和 31。
- prompt 必须逐 token 处理，并将每个 `present.*` 输出回填到对应的
  `past_key_values.*` 输入。

这与可以一次 prefill 完整 prompt 的普通 decoder-only 示例不同。实现沿用了
Mobius
[`examples/text_generation.py`](https://github.com/onnxruntime/mobius/blob/main/examples/text_generation.py)
的基本结构，同时增加了 Qwen3.5 所需的混合状态处理。

## 文件说明

| 文件 | 用途 |
|---|---|
| `Qwen3_5_4B_Hmm_GGUF_to_CPU_ONNX_Mobius_fixed.ipynb` | Colab 转换与验证流程 |
| `run_hmm_onnx.py` | 独立类型化决策运行脚本 |

Notebook 在 Colab 中使用 `/content/onnx_outputs`，在非 Colab 环境中使用
`.mobius_colab_run/onnx_outputs`。生成的模型、下载的 GGUF、虚拟环境、压缩包和
已执行 notebook 均不会提交到 Git，因为 ONNX 外部权重文件约为 2.6 GB。

## 可复现版本

| 组件 | 版本或 revision |
|---|---|
| GGUF 仓库 | `n4ze3m/Qwen3.5-4B-Hmm` |
| GGUF revision | `c27fa3c627dfaced343c6ba9d3a0d00243a3be51` |
| GGUF 文件 | `Qwen3.5-4B-Hmm-Q4_K_M.gguf` |
| GGUF SHA-256 | `5e03cb057049c56b421bd3c506d77fd8e0a77996bb8464148a02cfc3caac5229` |
| Mobius revision | `6b27a3f08b8b5d08ba9b14b416e3b435942bb0bd` |
| Tokenizer 仓库 | `Qwen/Qwen3.5-4B` |
| Tokenizer revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

Notebook 会从固定 Git revision 构建 Mobius wheel，而不是安装预编译的 Mobius
软件包。

## 快速开始

在高内存 Colab CPU runtime 中运行：

```text
Qwen3_5_4B_Hmm_GGUF_to_CPU_ONNX_Mobius_fixed.ipynb
```

运行已经转换好的本地模型：

```bash
python run_hmm_onnx.py \
  --model-dir .mobius_colab_run/onnx_outputs
```

从 JSON 文件读取请求：

```bash
python run_hmm_onnx.py \
  --model-dir .mobius_colab_run/onnx_outputs \
  --request request.json
```

`request.json` 示例：

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

## 决策类型

### `noul`

表示 yes/no 概率。返回的 `noul` 数值是 `true` 选项的归一化概率。

```json
{
  "type": "noul",
  "noul": 0.9745
}
```

### `choice`

返回概率最高的 key、全部归一化选项概率以及置信度。

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

返回数值等级的期望值、等级说明、归一化概率与置信度。

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

Hmm 使用 26 个选项字母训练。运行脚本通过每组 25 个选项并额外加入
`none_of_these` 的方式支持最多 255 个 choice 选项，与参考 Hmm server 的策略
一致。

## 已验证结果

生成的模型已经使用 ONNX Runtime 在 Apple Silicon CPU 上直接运行：

| 检查项 | 结果 |
|---|---|
| ONNX Runtime | `1.30.0` |
| Execution provider | `CPUExecutionProvider` |
| ONNX checker | 通过 |
| 有状态 prefill/decode | 通过 |
| Hmm 类型化决策测试 | 通过 |
| 模型计算图 | 618 KB |
| 外部模型数据 | 2.6 GB |
| 示例输入 token | 244 |
| 三问题推理时间 | 10.24 秒 |

实测示例结果：

- 紧急概率：`0.9745`
- 处理部门：`billing`
- Billing 概率：`0.7216`
- 优先级分数：`2.6653`，范围为 `0..3`

性能取决于 CPU、线程数、内存带宽、prompt 长度和 ONNX Runtime 版本。

## 重要限制

- Hmm 是实验性微调模型。其模型卡明确建议不要将它作为唯一安全检查，也不要用于
  重要决策。
- Hmm 复现的是 System One 类型化决策思想与请求风格；每个问题仍需执行一次
  模型，并未复现 Jev 的原生并行采样器。
- 输出类型由适配代码约束，底层 Qwen 模型本身不提供数学意义上的类型保证。
- 概率是选项字母概率在当前候选项上的归一化结果，应针对实际工作负载重新评估
  校准程度。
- Hmm 使用英语训练，主要面向不超过 768 token 的 prompt。
- Mobius 当前将 `qwen35` 的图构建和量化导入标记为 supported，但代表性真实权重
  runtime evidence 仍为 pending。因此，本项目使用 ONNX Runtime 对这个具体转换
  产物执行直接验证。
- `model.onnx` 和 `model.onnx.data` 必须放在同一个目录中。

## 参考资料与引用

主要参考资料：

1. TypeSafe AI，[Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
2. Muhammed Nazeem，[Hmm (Qwen3.5-4B)](https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm)
3. Microsoft，[ONNXRuntime Mobius](https://github.com/onnxruntime/mobius)
4. Microsoft，[Mobius GGUF 导入文档](https://onnxruntime.github.io/mobius/api/build_from_gguf.html)
5. Qwen Team，[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

使用转换后的模型时，请引用原始 Hmm 模型并注明使用了 Mobius 转换流程。Hmm 模型
卡提供了以下引用：

```bibtex
@misc{nazeem2026hmm,
  author = {Muhammed Nazeem},
  title  = {Hmm: a small open model for typed decisions},
  year   = {2026},
  url    = {https://github.com/n4ze3m/hmm}
}
```

本转换流程可使用以下建议引用：

```bibtex
@software{qwen35_hmm_mobius_onnx,
  title  = {Qwen3.5-4B-Hmm GGUF to CPU ONNX with ONNXRuntime Mobius},
  year   = {2026},
  note   = {Conversion and direct CPU inference workflow},
  url    = {https://github.com/onnxruntime/mobius}
}
```

如果本项目发布到代码仓库，请将第二条引用中的 URL 替换为实际项目地址。
