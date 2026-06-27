from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


PAD = "<pad>"
UNK = "<unk>"


@dataclass(frozen=True)
class CharTokenizer:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({char for text in texts for char in text})
        vocab = [PAD, UNK, *chars]
        return cls({token: idx for idx, token in enumerate(vocab)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(char, self.unk_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        return "".join(id_to_token.get(idx, "") for idx in ids if idx != self.pad_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))
