from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

from .data import build_dataset, dataset_path, read_jsonl, smoke_split_sizes
from .formats import Method, continuous_item, format_text
from .tasks import Example, verify_answer
from .tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny decoder on CLT synthetic tasks.")
    parser.add_argument("--task", default="graph_reachability")
    parser.add_argument("--method", choices=["direct", "cot", "masked_cot", "soft", "latent"], default="direct")
    parser.add_argument("--data-dir", type=Path, default=Path("data/phase1a_smoke"))
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu or npu:0")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    import torch

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_device(torch.device(args.device))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.build_data or not dataset_path(args.data_dir, "train", args.task).exists():
        build_dataset(args.data_dir, [args.task], smoke_split_sizes())

    train_examples = read_jsonl(dataset_path(args.data_dir, "train", args.task))
    dev_examples = read_jsonl(dataset_path(args.data_dir, "dev", args.task))
    id_examples = read_jsonl(dataset_path(args.data_dir, "id_test", args.task))
    ood_examples = read_jsonl(dataset_path(args.data_dir, "ood_test", args.task))

    tokenizer = build_tokenizer(train_examples + dev_examples + id_examples + ood_examples)
    from .tiny_model import TinyDecoder, TinyDecoderConfig

    model = TinyDecoder(
        TinyDecoderConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        example = random.choice(train_examples)
        loss = _loss_for_example(model, tokenizer, example, args.method, args.k, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, args.steps // 4) == 0:
            print({"step": step, "loss": round(losses[-1], 4)})

    metrics = {
        "task": args.task,
        "method": args.method,
        "steps": args.steps,
        "k": args.k if args.method in {"soft", "latent"} else None,
        "train_loss_last": losses[-1],
        "elapsed_sec": round(time.time() - start_time, 3),
        "dev": evaluate(model, tokenizer, dev_examples[: args.eval_examples], args.method, args.k, args.device, args.max_new_tokens),
        "id_test": evaluate(
            model, tokenizer, id_examples[: args.eval_examples], args.method, args.k, args.device, args.max_new_tokens
        ),
        "ood_test": evaluate(
            model, tokenizer, ood_examples[: args.eval_examples], args.method, args.k, args.device, args.max_new_tokens
        ),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def build_tokenizer(examples: list[Example]) -> CharTokenizer:
    texts = []
    for example in examples:
        for method in ["direct", "cot", "masked_cot"]:
            texts.append(format_text(example, method).text)
        item = continuous_item(example)
        texts.append(item.prefix + item.answer)
    return CharTokenizer.build(texts)


def _loss_for_example(model, tokenizer: CharTokenizer, example: Example, method: Method, k: int, device: str):
    import torch

    if method in {"direct", "cot", "masked_cot"}:
        item = format_text(example, method)
        ids = torch.tensor(tokenizer.encode(item.text), device=device, dtype=torch.long)
        input_ids = ids[:-1].unsqueeze(0)
        labels = ids[1:].unsqueeze(0)
        char_positions = torch.arange(1, ids.numel(), device=device)
        label_mask = (char_positions >= item.loss_start).unsqueeze(0)
        return model.text_loss(input_ids, labels, label_mask)

    item = continuous_item(example)
    prefix_ids = torch.tensor(tokenizer.encode(item.prefix), device=device, dtype=torch.long)
    answer_ids = torch.tensor(tokenizer.encode(item.answer), device=device, dtype=torch.long)
    return model.continuous_loss(prefix_ids, answer_ids, num_steps=k, mode=method)


def evaluate(model, tokenizer: CharTokenizer, examples: list[Example], method: Method, k: int, device: str, max_new_tokens: int) -> dict:
    import torch

    model.eval()
    correct = 0
    predictions = []
    with torch.no_grad():
        for example in examples:
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                prefix_ids = torch.tensor(tokenizer.encode(prefix), device=device, dtype=torch.long)
                generated = tokenizer.decode(model.generate_text(prefix_ids, max_new_tokens=max_new_tokens))
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                prefix_ids = torch.tensor(tokenizer.encode(item.prefix), device=device, dtype=torch.long)
                generated = tokenizer.decode(
                    model.generate_continuous(prefix_ids, num_steps=k, mode=method, max_new_tokens=max_new_tokens)
                )
            else:
                prefix = f"Problem:\n{example.prompt}\nReasoning: "
                prefix_ids = torch.tensor(tokenizer.encode(prefix), device=device, dtype=torch.long)
                generated = tokenizer.decode(model.generate_text(prefix_ids, max_new_tokens=max_new_tokens))

            answer = extract_answer(generated)
            ok = verify_answer(example, answer)
            correct += int(ok)
            if len(predictions) < 5:
                predictions.append({"expected": example.answer, "generated": generated[:120], "parsed": answer, "ok": ok})
    model.train()
    return {"accuracy": correct / max(1, len(examples)), "num_examples": len(examples), "samples": predictions}


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


if __name__ == "__main__":
    main()
