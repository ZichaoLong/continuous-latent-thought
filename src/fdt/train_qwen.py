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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen/Hugging Face causal LM on FDT tasks.")
    parser.add_argument("--model-name-or-path", default=os.environ.get("MODEL_NAME_OR_PATH", "Qwen/Qwen3-0.6B-Base"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--task", default="graph_reachability")
    parser.add_argument("--method", choices=["direct", "cot", "masked_cot", "soft", "latent"], default="direct")
    parser.add_argument("--data-dir", type=Path, default=Path("data/qwen_smoke"))
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--difficulty", choices=["standard", "easy", "easy_ladder", "simple"], default="easy_ladder")
    parser.add_argument("--device", default="cpu", help="cpu or npu:0")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--eval-mode", choices=["generate", "binary_choice"], default="binary_choice")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--case-examples", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    import torch

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_device(torch.device(args.device))
        if args.dtype == "float16" and args.steps > 0:
            print(
                "Warning: full-parameter FP16 AdamW on NPU can produce NaNs; prefer --dtype bfloat16 for Qwen SFT.",
                flush=True,
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.build_data or not dataset_path(args.data_dir, "train", args.task).exists():
        build_dataset(args.data_dir, [args.task], smoke_split_sizes(), difficulty=args.difficulty)

    train_examples = read_jsonl(dataset_path(args.data_dir, "train", args.task))
    dev_examples = read_jsonl(dataset_path(args.data_dir, "dev", args.task))
    id_examples = read_jsonl(dataset_path(args.data_dir, "id_test", args.task))
    ood_examples = read_jsonl(dataset_path(args.data_dir, "ood_test", args.task))

    tokenizer, model = load_model(args)
    model = model.to(args.device)
    model.train()

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()

    if args.freeze_backbone:
        for parameter in model.model.parameters():
            parameter.requires_grad_(False)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain. Disable --freeze-backbone or use a trainable adapter.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    start_time = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        example = random.choice(train_examples)
        loss = loss_for_example(model, tokenizer, example, args.method, args.k, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, args.steps // 4) == 0:
            print({"step": step, "loss": round(losses[-1], 4)}, flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_name_or_path": args.model_name_or_path,
                "task": args.task,
                "method": args.method,
                "difficulty": args.difficulty,
                "k": args.k if args.method in {"soft", "latent"} else None,
                "steps": args.steps,
                "seed": args.seed,
            },
            args.save_checkpoint,
        )

    metrics = {
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "method": args.method,
        "difficulty": args.difficulty,
        "eval_mode": args.eval_mode,
        "steps": args.steps,
        "k": args.k if args.method in {"soft", "latent"} else None,
        "train_loss_last": losses[-1] if losses else None,
        "elapsed_sec": round(time.time() - start_time, 3),
        "checkpoint_saved": str(args.save_checkpoint) if args.save_checkpoint is not None else None,
        "dev": evaluate(
            model,
            tokenizer,
            dev_examples[: args.eval_examples],
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        ),
        "id_test": evaluate(
            model,
            tokenizer,
            id_examples[: args.eval_examples],
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        ),
        "ood_test": evaluate(
            model,
            tokenizer,
            ood_examples[: args.eval_examples],
            args.method,
            args.k,
            args.device,
            args.max_new_tokens,
            args.eval_mode,
            args.case_examples,
        ),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return tokenizer, HFContinuousWrapper(model, soft_temperature=args.soft_temperature)


def HFContinuousWrapper(model, soft_temperature: float = 1.0):
    """Create a continuous-thinking adapter after torch has been imported."""

    import torch.nn as nn

    class _HFContinuousWrapper(nn.Module):
        def __init__(self, inner_model, temperature: float) -> None:
            super().__init__()
            self.model = inner_model
            self.soft_temperature = temperature
            hidden_size = getattr(inner_model.config, "hidden_size", None) or getattr(inner_model.config, "n_embd")
            self.latent_norm = nn.LayerNorm(hidden_size)
            self.latent_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.eye_(self.latent_proj.weight)
            nn.init.zeros_(self.latent_proj.bias)

        def continuous_loss(self, prefix_ids, answer_ids, num_steps: int, mode: Method):
            import torch.nn.functional as F

            logits = self._continuous_logits(prefix_ids, answer_ids[:-1], num_steps, mode)
            start = prefix_ids.numel() + num_steps - 1
            end = start + answer_ids.numel()
            supervised_logits = logits[:, start:end, :].squeeze(0).float()
            return F.cross_entropy(supervised_logits, answer_ids)

        def continuous_candidate_nll(self, prefix_ids, candidate_ids, num_steps: int, mode: Method):
            import torch.nn.functional as F

            logits = self._continuous_logits(prefix_ids, candidate_ids[:-1], num_steps, mode).squeeze(0)
            start = prefix_ids.numel() + num_steps - 1
            end = start + candidate_ids.numel()
            token_losses = F.cross_entropy(logits[start:end, :].float(), candidate_ids, reduction="none")
            return token_losses.mean()

        def generate_continuous(
            self,
            prefix_ids,
            num_steps: int,
            mode: Method,
            max_new_tokens: int,
            eos_token_id: int | None,
        ):
            import torch

            seq_embeds = self.model.get_input_embeddings()(prefix_ids.unsqueeze(0))
            seq_embeds = self._append_continuous_steps(seq_embeds, num_steps, mode)

            generated = []
            for _ in range(max_new_tokens):
                outputs = self.model(inputs_embeds=seq_embeds, use_cache=False)
                next_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                generated.append(next_id)
                next_id_tensor = torch.tensor([[next_id]], device=seq_embeds.device, dtype=prefix_ids.dtype)
                next_embed = self.model.get_input_embeddings()(next_id_tensor)
                seq_embeds = torch.cat([seq_embeds, next_embed], dim=1)
                if eos_token_id is not None and next_id == eos_token_id:
                    break
            return generated

        def _continuous_logits(self, prefix_ids, suffix_input_ids, num_steps: int, mode: Method):
            import torch

            seq_embeds = self.model.get_input_embeddings()(prefix_ids.unsqueeze(0))
            seq_embeds = self._append_continuous_steps(seq_embeds, num_steps, mode)
            if suffix_input_ids.numel() > 0:
                suffix_embeds = self.model.get_input_embeddings()(suffix_input_ids.unsqueeze(0))
                seq_embeds = torch.cat([seq_embeds, suffix_embeds], dim=1)
            return self.model(inputs_embeds=seq_embeds, use_cache=False).logits

        def _append_continuous_steps(self, seq_embeds, num_steps: int, mode: Method):
            import torch

            for _ in range(num_steps):
                outputs = self.model(inputs_embeds=seq_embeds, output_hidden_states=True, use_cache=False)
                next_embed = self._next_continuous_embed(outputs, mode)
                seq_embeds = torch.cat([seq_embeds, next_embed.unsqueeze(1)], dim=1)
            return seq_embeds

        def _next_continuous_embed(self, outputs, mode: Method):
            import torch

            if mode == "latent":
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                return self.latent_proj(self.latent_norm(last_hidden))
            if mode == "soft":
                temperature = max(self.soft_temperature, 1e-6)
                probs = torch.softmax(outputs.logits[:, -1, :].float() / temperature, dim=-1)
                embedding_weight = self.model.get_input_embeddings().weight
                return probs.to(embedding_weight.dtype) @ embedding_weight
            raise ValueError(f"Unknown continuous mode: {mode}")

    return _HFContinuousWrapper(model, soft_temperature)


def loss_for_example(model, tokenizer, example: Example, method: Method, k: int, device: str):
    import torch

    if method in {"direct", "cot", "masked_cot"}:
        item = format_text(example, method)
        encoded = encode_with_offsets(tokenizer, item.text)
        ids = torch.tensor(encoded["input_ids"], device=device, dtype=torch.long)
        labels = ids.clone()
        loss_mask = loss_mask_from_offsets(encoded, item.loss_start, device)
        if loss_mask is None:
            prefix_len = len(tokenizer(item.text[: item.loss_start], add_special_tokens=False)["input_ids"])
            labels[:prefix_len] = -100
        else:
            labels[loss_mask] = -100
        outputs = model.model(input_ids=ids.unsqueeze(0), use_cache=False)
        return causal_lm_loss(outputs.logits, labels.unsqueeze(0))

    item = continuous_item(example)
    prefix_ids = encode(tokenizer, item.prefix, device)
    answer_ids = encode(tokenizer, item.answer, device)
    return model.continuous_loss(prefix_ids, answer_ids, num_steps=k, mode=method)


def evaluate(model, tokenizer, examples, method, k, device, max_new_tokens, eval_mode, case_examples):
    model.eval()
    if eval_mode == "binary_choice" and all(example.answer in {"YES", "NO"} for example in examples):
        result = evaluate_binary_choice(model, tokenizer, examples, method, k, device, max_new_tokens, case_examples)
        model.train()
        return result

    import torch

    correct = 0
    predictions = []
    cases = {"success": [], "failure": []}
    with torch.no_grad():
        for example in examples:
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                generated = generate_text(model.model, tokenizer, encode(tokenizer, prefix, device), max_new_tokens)
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                generated_ids = model.generate_continuous(
                    encode(tokenizer, item.prefix, device),
                    num_steps=k,
                    mode=method,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                prefix = f"Problem:\n{example.prompt}\nReasoning: "
                generated = generate_text(model.model, tokenizer, encode(tokenizer, prefix, device), max_new_tokens)

            answer = extract_answer(generated)
            ok = verify_answer(example, answer)
            correct += int(ok)
            sample = {"expected": example.answer, "generated": generated[:160], "parsed": answer, "ok": ok}
            if len(predictions) < 5:
                predictions.append(sample)
            record_case(cases, {**sample, "prompt": example.prompt[:500], "metadata": example.metadata}, ok, case_examples)
    model.train()
    return {"accuracy": correct / max(1, len(examples)), "num_examples": len(examples), "samples": predictions, "cases": cases}


def evaluate_binary_choice(model, tokenizer, examples, method, k, device, max_trace_tokens, case_examples):
    import torch

    choices = ["YES", "NO"]
    candidate_ids = {choice: encode(tokenizer, f"{choice}\n", device) for choice in choices}
    correct = 0
    predictions = []
    cases = {"success": [], "failure": []}
    with torch.no_grad():
        for example in examples:
            generated_trace = None
            if method == "direct":
                prefix = f"Problem:\n{example.prompt}\nAnswer: "
                scores = {
                    choice: float(candidate_nll(model.model, tokenizer, prefix, ids, device).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }
            elif method in {"soft", "latent"}:
                item = continuous_item(example)
                prefix_ids = encode(tokenizer, item.prefix, device)
                scores = {
                    choice: float(model.continuous_candidate_nll(prefix_ids, ids, num_steps=k, mode=method).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }
            else:
                trace_prefix = f"Problem:\n{example.prompt}\nReasoning: "
                generated_trace = generate_text(model.model, tokenizer, encode(tokenizer, trace_prefix, device), max_trace_tokens)
                generated_trace = generated_trace.split("Answer:", 1)[0].rstrip()
                answer_prefix = f"{trace_prefix}{generated_trace}\nAnswer: "
                scores = {
                    choice: float(candidate_nll(model.model, tokenizer, answer_prefix, ids, device).detach().cpu())
                    for choice, ids in candidate_ids.items()
                }

            answer = min(scores, key=scores.get)
            ok = verify_answer(example, answer)
            correct += int(ok)
            sample = {"expected": example.answer, "parsed": answer, "scores": scores, "ok": ok}
            if generated_trace is not None:
                sample["generated_trace"] = generated_trace[:160]
            if len(predictions) < 5:
                predictions.append(sample)
            record_case(cases, {**sample, "prompt": example.prompt[:500], "metadata": example.metadata}, ok, case_examples)
    return {"accuracy": correct / max(1, len(examples)), "num_examples": len(examples), "samples": predictions, "cases": cases}


def candidate_nll(model, tokenizer, prefix: str, candidate_ids, device: str):
    ids = encode(tokenizer, prefix, device)
    input_ids = concat_ids(ids, candidate_ids)
    labels = input_ids.clone()
    labels[: ids.numel()] = -100
    outputs = model(input_ids=input_ids.unsqueeze(0), use_cache=False)
    return causal_lm_loss(outputs.logits, labels.unsqueeze(0))


def causal_lm_loss(logits, labels):
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != -100
    if not bool(valid.any()):
        raise ValueError("No supervised tokens available for causal LM loss.")
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return losses[valid.reshape(-1)].mean()


def generate_text(model, tokenizer, prefix_ids, max_new_tokens: int) -> str:
    output_ids = model.generate(
        input_ids=prefix_ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0, prefix_ids.numel() :]
    return tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)


def encode(tokenizer, text: str, device: str):
    import torch

    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], device=device, dtype=torch.long)


def encode_with_offsets(tokenizer, text: str) -> dict:
    try:
        return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError:
        return tokenizer(text, add_special_tokens=False)


def loss_mask_from_offsets(encoded: dict, loss_start: int, device: str):
    import torch

    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    return torch.tensor([end <= loss_start for _, end in offsets], device=device, dtype=torch.bool)


def concat_ids(left, right):
    import torch

    return torch.cat([left, right], dim=0)


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def record_case(cases: dict, sample: dict, ok: bool, limit: int) -> None:
    if limit <= 0:
        return
    bucket = "success" if ok else "failure"
    if len(cases[bucket]) < limit:
        cases[bucket].append(sample)


if __name__ == "__main__":
    main()
