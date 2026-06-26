from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .tasks import Example


Method = Literal["direct", "cot", "masked_cot", "soft", "latent"]


@dataclass(frozen=True)
class TextItem:
    text: str
    loss_start: int
    answer: str


@dataclass(frozen=True)
class ContinuousItem:
    prefix: str
    answer: str


def problem_prefix(example: Example) -> str:
    return f"Problem:\n{example.prompt}\n"


def direct_item(example: Example) -> TextItem:
    prefix = f"{problem_prefix(example)}Answer: "
    text = f"{prefix}{example.answer}\n"
    return TextItem(text=text, loss_start=len(prefix), answer=example.answer)


def cot_item(example: Example) -> TextItem:
    prefix = f"{problem_prefix(example)}Reasoning: "
    text = f"{prefix}{example.trace}\nAnswer: {example.answer}\n"
    return TextItem(text=text, loss_start=len(prefix), answer=example.answer)


def masked_cot_item(example: Example) -> TextItem:
    prefix = f"{problem_prefix(example)}Reasoning: {example.trace}\nAnswer: "
    text = f"{prefix}{example.answer}\n"
    return TextItem(text=text, loss_start=len(prefix), answer=example.answer)


def continuous_item(example: Example) -> ContinuousItem:
    return ContinuousItem(prefix=f"{problem_prefix(example)}Answer: ", answer=f"{example.answer}\n")


def format_text(example: Example, method: Method) -> TextItem:
    if method == "direct":
        return direct_item(example)
    if method == "cot":
        return cot_item(example)
    if method == "masked_cot":
        return masked_cot_item(example)
    raise ValueError(f"{method} does not produce a pure text training item")


def format_continuous(example: Example, method: Method) -> ContinuousItem:
    if method in {"soft", "latent"}:
        return continuous_item(example)
    raise ValueError(f"{method} does not produce a continuous training item")
