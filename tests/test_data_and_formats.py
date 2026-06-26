from clt.data import build_split, read_jsonl
from clt.formats import continuous_item, format_text
from clt.tasks import generate_example
from clt.tokenizer import CharTokenizer


def test_build_split_writes_jsonl(tmp_path):
    path = build_split(tmp_path, "graph_reachability", "train", 3, seed_start=10)
    examples = read_jsonl(path)
    assert len(examples) == 3
    assert examples[0].metadata["seed"] == 10


def test_build_split_accepts_easy_difficulty(tmp_path):
    path = build_split(tmp_path, "graph_reachability", "train", 3, seed_start=10, difficulty="easy")
    examples = read_jsonl(path)
    assert len(examples) == 3
    assert all(example.metadata["difficulty"] == "easy" for example in examples)


def test_training_formats_have_expected_loss_boundaries():
    example = generate_example("graph_reachability", seed=0, split="train")

    direct = format_text(example, "direct")
    assert direct.text[direct.loss_start :].startswith(example.answer)

    cot = format_text(example, "cot")
    assert cot.text[cot.loss_start :].startswith(example.trace)
    assert f"Answer: {example.answer}" in cot.text

    masked = format_text(example, "masked_cot")
    assert masked.text[masked.loss_start :].startswith(example.answer)
    assert example.trace in masked.text[: masked.loss_start]

    latent = continuous_item(example)
    assert latent.prefix.endswith("Answer: ")
    assert latent.answer.startswith(example.answer)


def test_char_tokenizer_round_trips_text():
    tokenizer = CharTokenizer.build(["abc", "bcd"])
    text = "abcd"
    assert tokenizer.decode(tokenizer.encode(text)) == text
