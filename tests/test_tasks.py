import json

from clt.tasks import generate_example, list_tasks, parse_arithmetic_answer, verify_answer


def test_generators_are_deterministic():
    for task in list_tasks():
        first = generate_example(task, seed=123, split="train")
        second = generate_example(task, seed=123, split="train")
        assert first == second


def test_examples_are_json_serializable_and_self_verifying():
    for task in list_tasks():
        for split in ["train", "dev", "id_test", "ood_test"]:
            example = generate_example(task, seed=42, split=split)
            payload = json.loads(example.to_json())
            assert payload["prompt"]
            assert payload["trace"]
            assert payload["answer"]
            assert payload["metadata"]["task"] == task
            assert verify_answer(example, example.answer)


def test_ood_examples_are_larger_or_deeper():
    graph_train = generate_example("graph_reachability", seed=1, split="train")
    graph_ood = generate_example("graph_reachability", seed=1, split="ood_test")
    assert graph_ood.metadata["num_nodes"] >= graph_train.metadata["num_nodes"]

    expr_train = generate_example("symbolic_arithmetic", seed=1, split="train")
    expr_ood = generate_example("symbolic_arithmetic", seed=1, split="ood_test")
    assert expr_ood.metadata["max_depth"] > expr_train.metadata["max_depth"]


def test_arithmetic_answer_parser_accepts_simple_expressions():
    assert parse_arithmetic_answer("Answer: (3+5)-2") == 6
    assert parse_arithmetic_answer("not arithmetic") is None
