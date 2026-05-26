import json
from pathlib import Path

import pytest

from src.classes.Config import Config
from src.parsing import (
    get_function_definition,
    get_input,
    get_output_file,
    test_args,
)
from src.process_data import process_data

# ----------------------------
# Helpers
# ----------------------------


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file)


# ----------------------------
# Config tests
# ----------------------------


def test_config_valid() -> None:
    config = Config(
        function_definition=[
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            }
        ],
        input=[{"prompt": "What is the sum of 2 and 3?"}],
        output_file="data/output/function_calls.json",
    )

    assert config.output_file == "data/output/function_calls.json"
    assert config.input[0]["prompt"] == "What is the sum of 2 and 3?"


def test_config_invalid_function_definition() -> None:
    with pytest.raises(ValueError):
        Config(
            function_definition=[
                {
                    "name": "fn_add_numbers",
                    # missing description
                    "parameters": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "returns": {"type": "number"},
                }
            ],
            input=[{"prompt": "test"}],
            output_file="out.json",
        )


def test_config_invalid_input() -> None:
    with pytest.raises(ValueError):
        Config(
            function_definition=[
                {
                    "name": "fn_add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "returns": {"type": "number"},
                }
            ],
            input=[{"wrong_key": "test"}],
            output_file="out.json",
        )


# ----------------------------
# parsing.py tests
# ----------------------------


def test_get_output_file_default() -> None:
    argv = ["prog"]
    assert get_output_file(argv) == "data/output/function_calls.json"


def test_get_output_file_custom() -> None:
    argv = ["prog", "--output", "custom.json"]
    # this test may fail with your current implementation
    assert get_output_file(argv) == "custom.json"


def test_test_args_valid() -> None:
    argv = [
        "prog",
        "--function_definition",
        "functions.json",
        "--input",
        "input.json",
        "--output",
        "output.json",
    ]
    test_args(argv)


def test_test_args_duplicate() -> None:
    argv = [
        "prog",
        "--input",
        "a.json",
        "--input",
        "b.json",
    ]
    with pytest.raises(Exception):
        test_args(argv)


def test_test_args_unknown() -> None:
    argv = ["prog", "--unknown", "x"]
    with pytest.raises(Exception):
        test_args(argv)


def test_get_function_definition_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path
    file_path = fake_root / "data" / "input" / "functions_definition.json"

    write_json(
        file_path,
        [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            }
        ],
    )

    monkeypatch.chdir(fake_root)

    result = get_function_definition(["prog"])
    assert result[0]["name"] == "fn_add_numbers"


def test_get_input_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path
    file_path = fake_root / "data" / "input" / "function_calling_tests.json"

    write_json(file_path, [{"prompt": "hello"}])

    monkeypatch.chdir(fake_root)

    result = get_input(["prog"])
    assert result == [{"prompt": "hello"}]


def test_get_input_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        get_input(["prog"])


def test_get_input_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_path = tmp_path / "data" / "input" / "function_calling_tests.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("{ invalid json", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(json.JSONDecodeError):
        get_input(["prog"])


# ----------------------------
# process_data tests
# ----------------------------


def test_process_data_writes_expected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_file = tmp_path / "function_calls.json"

    config = Config(
        function_definition=[
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            }
        ],
        input=[{"prompt": "What is the sum of 2 and 3?"}],
        output_file=str(output_file),
    )

    class FakePromptSolver:
        def __init__(self, config: Config) -> None:
            self.config = config

        def get_fn_name(self, prompt: str) -> str:
            return "fn_add_numbers"

        def get_fn_parameters(
            self, fn_name: str, prompt: str
        ) -> dict[str, float]:
            return {"a": 2.0, "b": 3.0}

    monkeypatch.setattr("src.process_data.PromptSolver", FakePromptSolver)

    process_data(config)

    assert output_file.exists()

    content = json.loads(output_file.read_text(encoding="utf-8"))
    assert content == [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_process_data_skips_prompt_on_solver_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_file = tmp_path / "function_calls.json"

    config = Config(
        function_definition=[
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            }
        ],
        input=[
            {"prompt": "good prompt"},
            {"prompt": "bad prompt"},
        ],
        output_file=str(output_file),
    )

    class FakePromptSolver:
        def __init__(self, config: Config) -> None:
            self.config = config

        def get_fn_name(self, prompt: str) -> str:
            if prompt == "bad prompt":
                raise ValueError("solver failed")
            return "fn_add_numbers"

        def get_fn_parameters(
            self, fn_name: str, prompt: str
        ) -> dict[str, float]:
            return {"a": 1.0, "b": 2.0}

    monkeypatch.setattr("src.process_data.PromptSolver", FakePromptSolver)

    process_data(config)

    captured = capsys.readouterr()
    assert "solver failed" in captured.out

    content = json.loads(output_file.read_text(encoding="utf-8"))
    assert content == [
        {
            "prompt": "good prompt",
            "name": "fn_add_numbers",
            "parameters": {"a": 1.0, "b": 2.0},
        }
    ]
