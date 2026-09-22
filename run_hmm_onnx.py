from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = 255
TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def options_for(name: str, question: dict[str, Any]) -> list[tuple[str, str]]:
    if not isinstance(question, dict) or question.get("instructions") is None:
        raise ValueError(f'question "{name}" needs "type" and "instructions"')

    question_type = question.get("type")
    criteria = question.get("criteria")

    if question_type == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(f'choice "{name}" needs 2-{MAX_OPTIONS} options')
        return [(str(key), text(value)) for key, value in criteria.items()]

    if question_type == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f'score "{name}" needs 2-10 ordered levels')
        return [(str(index), text(value)) for index, value in enumerate(criteria)]

    if question_type == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        return [
            ("false", text(criteria.get("false", "No"))),
            ("true", text(criteria.get("true", "Yes"))),
        ]

    raise ValueError(f'question "{name}" has unknown type {question_type!r}')


def build_prompt(
    state: Any,
    question: dict[str, Any],
    options: list[tuple[str, str]],
) -> str:
    lines = [
        f"{LETTERS[index]}: {key} — {description}"
        for index, (key, description) in enumerate(options)
    ]
    user = (
        "State (data to evaluate):\n"
        + text(state)
        + "\n\nQuestion:\n"
        + text(question["instructions"])
        + "\n\nOptions:\n"
        + "\n".join(lines)
        + "\nReturn only the option letter."
    )
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


class HmmOnnx:
    def __init__(self, model_dir: Path) -> None:
        model_path = model_dir / "model.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = False
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.letter_token_ids = {}
        for letter in LETTERS:
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Option letter {letter!r} is not one token: {token_ids}")
            self.letter_token_ids[letter] = token_ids[0]

    @staticmethod
    def _state_shape(model_input: Any) -> tuple[int, ...]:
        shape = []
        for axis, dimension in enumerate(model_input.shape):
            if isinstance(dimension, int):
                shape.append(dimension)
            elif (
                model_input.name.endswith((".key", ".value"))
                and axis == 2
            ):
                shape.append(0)
            else:
                shape.append(1)
        return tuple(shape)

    def _initial_states(self) -> dict[str, np.ndarray]:
        states = {}
        for model_input in self.session.get_inputs():
            if not model_input.name.startswith("past_key_values."):
                continue
            dtype = ORT_TO_NUMPY.get(model_input.type)
            if dtype is None:
                raise TypeError(f"Unsupported state dtype: {model_input.type}")
            states[model_input.name] = np.zeros(
                self._state_shape(model_input),
                dtype=dtype,
            )
        return states

    def _forward_prompt(self, prompt: str) -> tuple[np.ndarray, int, float]:
        input_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="np",
        )["input_ids"].astype(np.int64)

        states = self._initial_states()
        outputs = None
        started = time.perf_counter()

        for position, token_id in enumerate(input_ids[0]):
            feeds = {
                "input_ids": np.asarray([[token_id]], dtype=np.int64),
                "attention_mask": np.ones((1, position + 1), dtype=np.int64),
                "position_ids": np.asarray([[position]], dtype=np.int64),
                **states,
            }
            missing = self.input_names - feeds.keys()
            if missing:
                raise ValueError(f"Missing model inputs: {sorted(missing)}")

            values = self.session.run(
                self.output_names,
                {name: feeds[name] for name in self.input_names},
            )
            outputs = dict(zip(self.output_names, values, strict=True))
            states = {
                name: outputs[
                    f"present.{name.removeprefix('past_key_values.')}"
                ]
                for name in states
            }

        if outputs is None:
            raise ValueError("Prompt produced no input tokens")

        logits = outputs["logits"][0, -1].astype(np.float64)
        if not np.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits detected")

        return logits, int(input_ids.shape[1]), time.perf_counter() - started

    def _letter_probabilities(
        self,
        prompt: str,
        letters: str,
    ) -> tuple[dict[str, float], int, float]:
        logits, input_tokens, elapsed = self._forward_prompt(prompt)
        shifted = logits - np.max(logits)
        denominator = float(np.exp(shifted).sum())
        probabilities = {
            letter: float(
                np.exp(shifted[self.letter_token_ids[letter]]) / denominator
            )
            for letter in letters
        }
        return probabilities, input_tokens, elapsed

    @staticmethod
    def _round4(value: float) -> float:
        return math.floor(value * 10000 + 0.5) / 10000

    def answer(
        self,
        state: Any,
        name: str,
        question: dict[str, Any],
    ) -> dict[str, Any]:
        options = options_for(name, question)
        raw_probabilities = []
        input_tokens = 0
        elapsed = 0.0

        if len(options) <= len(LETTERS):
            letters = LETTERS[: len(options)]
            raw, count, seconds = self._letter_probabilities(
                build_prompt(state, question, options),
                letters,
            )
            raw_probabilities = [raw[letter] for letter in letters]
            input_tokens += count
            elapsed += seconds
        else:
            chunk_size = len(LETTERS) - 1
            none_option = ("none_of_these", "None of the other options fits")
            for start in range(0, len(options), chunk_size):
                chunk = options[start : start + chunk_size]
                chunk_options = [*chunk, none_option]
                letters = LETTERS[: len(chunk_options)]
                raw, count, seconds = self._letter_probabilities(
                    build_prompt(state, question, chunk_options),
                    letters,
                )
                raw_probabilities.extend(
                    raw[LETTERS[index]] for index in range(len(chunk))
                )
                input_tokens += count
                elapsed += seconds

        total = sum(raw_probabilities)
        probabilities = (
            [value / total for value in raw_probabilities]
            if total > 0
            else [1.0 / len(options)] * len(options)
        )
        keys = [key for key, _ in options]
        best = int(np.argmax(probabilities))
        probability_map = {
            key: self._round4(probabilities[index])
            for index, key in enumerate(keys)
        }
        confidence = self._round4(
            (len(options) * probabilities[best] - 1) / (len(options) - 1)
        )

        question_type = question["type"]
        if question_type == "noul":
            result = {
                "type": "noul",
                "noul": self._round4(probabilities[1]),
            }
        elif question_type == "choice":
            result = {
                "type": "choice",
                "choice": keys[best],
                "probabilities": probability_map,
                "confidence": confidence,
            }
        else:
            result = {
                "type": "score",
                "score": self._round4(
                    sum(index * probability for index, probability in enumerate(probabilities))
                ),
                "legend": dict(options),
                "probabilities": probability_map,
                "confidence": confidence,
            }

        return {
            "input_tokens": input_tokens,
            "seconds": elapsed,
            "result": result,
        }

    def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("state") in (None, ""):
            raise ValueError('"state" is required')
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError('"questions" must be a non-empty object')

        started = time.perf_counter()
        answers = {}
        input_tokens = 0
        inference_seconds = 0.0

        for name, question in questions.items():
            answer = self.answer(body["state"], name, question)
            answers[name] = answer["result"]
            input_tokens += answer["input_tokens"]
            inference_seconds += answer["seconds"]

        return {
            "model": "Qwen3.5-4B-Hmm-Q4_K_M.onnx",
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "inference_seconds": round(inference_seconds, 4),
        }


DEFAULT_REQUEST = {
    "state": "Help! My payouts have failed for 3 days. I need the money today.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".mobius_colab_run/onnx_outputs"),
    )
    parser.add_argument(
        "--request",
        type=Path,
        help="Optional JSON request file. Uses the model-card example by default.",
    )
    args = parser.parse_args()

    request = (
        json.loads(args.request.read_text(encoding="utf-8"))
        if args.request
        else DEFAULT_REQUEST
    )
    model = HmmOnnx(args.model_dir)
    print(json.dumps(model.decide(request), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
