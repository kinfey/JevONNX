import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_sample(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for current, line in enumerate(stream):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Sample index {index} is outside {path}")


def generate(model_path: Path, prompt: str, max_new_tokens: int) -> str:
    model = og.Model(str(model_path))
    tokenizer = og.Tokenizer(model)
    input_ids = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + max_new_tokens, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)
    while not generator.is_done():
        generator.generate_next_token()
    sequence = np.asarray(generator.get_sequence(0), dtype=np.int32)
    return tokenizer.decode(sequence[len(input_ids):]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Olive-exported Qwen3.5 ONNX model.")
    parser.add_argument("--model", type=Path, required=True, help="Directory containing genai_config.json.")
    parser.add_argument("--data", type=Path, help="Prepared validation JSONL containing prompt and label.")
    parser.add_argument("--index", type=int, default=0, help="Validation sample index.")
    parser.add_argument("--prompt", help="Already chat-templated prompt; skips gold-label checking.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="Exit non-zero when prediction differs from gold.")
    args = parser.parse_args()
    if not (args.model / "genai_config.json").is_file():
        raise FileNotFoundError(f"Missing genai_config.json in {args.model}")
    if args.prompt is None and args.data is None:
        parser.error("provide --prompt or --data")
    expected = None
    n_options = len(LETTERS)
    prompt = args.prompt
    if prompt is None:
        sample = load_sample(args.data, args.index)
        prompt = sample["prompt"]
        expected = LETTERS[sample["label"]]
        n_options = sample["n_options"]
    output = generate(args.model, prompt, args.max_new_tokens)
    match = re.search(rf"\b([{LETTERS[:n_options]}])\b", output.upper())
    prediction = match.group(1) if match else None
    result = {"output": output, "prediction": prediction, "expected": expected, "correct": expected is None or prediction == expected}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check and expected is not None and prediction != expected:
        raise AssertionError(f"Prediction {prediction!r} does not match expected {expected!r}")


if __name__ == "__main__":
    main()
