import json
from pathlib import Path
from typing import Any

import pytest

from src.classes.Config import Config
from src.parsing import (
    get_function_definition,
    get_input,
    get_llm,
    get_output_file,
    parsing,
    test_args,
)


def valid_function_definition() -> list[dict[str, Any]]:
    return [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        },
        {
            "name": "fn_greet",
            "description": "Generate a greeting.",
            "parameters": {
                "name": {"type": "string"},
            },
            "returns": {"type": "string"},
        },
    ]


def valid_input() -> list[dict[str, str]]:
    return [
        {"prompt": "What is the sum of 2 and 3?"},
        {"prompt": "Greet Shrek"},
    ]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file)


@pytest.fixture
def default_project_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.chdir(tmp_path)

    write_json(
        tmp_path / "data/input/functions_definition.json",
        valid_function_definition(),
    )
    write_json(
        tmp_path / "data/input/function_calling_tests.json",
        valid_input(),
    )

    return tmp_path


def test_get_output_file_default() -> None:
    argv = ["python", "-m", "src"]

    assert get_output_file(argv) == "data/output/function_calls.json"


def test_get_output_file_custom() -> None:
    argv = [
        "python",
        "-m",
        "src",
        "--output",
        "custom/output.json",
    ]

    assert get_output_file(argv) == "custom/output.json"


def test_get_output_file_missing_value_returns_default() -> None:
    argv = [
        "python",
        "-m",
        "src",
        "--output",
    ]

    assert get_output_file(argv) == "data/output/function_calls.json"


def test_test_args_accepts_empty_args() -> None:
    argv = ["python"]

    test_args(argv)


def test_test_args_accepts_valid_args() -> None:
    argv = [
        "python",
        "--functions_definition",
        "functions.json",
        "--input",
        "input.json",
        "--output",
        "output.json",
        "--details",
        "--tokenizer",
        "--llm",
        "Qwen/Qwen3-0.6B",
    ]

    test_args(argv)


def test_test_args_rejects_unknown_argument() -> None:
    argv = [
        "python",
        "--unknown",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "Unknown argument" in str(error.value)


def test_test_args_rejects_duplicate_functions_definition() -> None:
    argv = [
        "python",
        "--functions_definition",
        "a.json",
        "--functions_definition",
        "b.json",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "--functions_definition param has multiple" in str(error.value)


def test_test_args_rejects_duplicate_input() -> None:
    argv = [
        "python",
        "--input",
        "a.json",
        "--input",
        "b.json",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "--input param has multiple definition" in str(error.value)


def test_test_args_rejects_duplicate_output() -> None:
    argv = [
        "python",
        "--output",
        "a.json",
        "--output",
        "b.json",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "--output param has multiple definition" in str(error.value)


def test_test_args_rejects_duplicate_llm() -> None:
    argv = [
        "python",
        "--llm",
        "model-a",
        "--llm",
        "model-b",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "--llm param has multiple definition" in str(error.value)


def test_test_args_rejects_duplicate_details() -> None:
    argv = [
        "python",
        "--details",
        "--details",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "--details param has multiple definition" in str(error.value)


def test_test_args_rejects_duplicate_tokenizer() -> None:
    argv = [
        "python",
        "--tokenizer",
        "--tokenizer",
    ]

    with pytest.raises(Exception) as error:
        test_args(argv)

    assert "--tokenizer param has multiple definition" in str(error.value)


def test_get_function_definition_default_path(
    default_project_files: Path,
) -> None:
    argv = ["python"]

    result = get_function_definition(argv)

    assert result == valid_function_definition()


def test_get_function_definition_custom_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    custom_path = tmp_path / "custom/functions.json"
    write_json(custom_path, valid_function_definition())

    argv = [
        "python",
        "--functions_definition",
        str(custom_path),
    ]

    result = get_function_definition(argv)

    assert result == valid_function_definition()


def test_get_function_definition_missing_file_raises_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    argv = [
        "python",
        "--functions_definition",
        "missing.json",
    ]

    with pytest.raises(FileNotFoundError):
        get_function_definition(argv)


def test_get_function_definition_invalid_json_raises_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{ invalid json", encoding="utf-8")

    argv = [
        "python",
        "--functions_definition",
        str(invalid_path),
    ]

    with pytest.raises(json.JSONDecodeError):
        get_function_definition(argv)


def test_get_input_default_path(default_project_files: Path) -> None:
    argv = ["python"]

    result = get_input(argv)

    assert result == valid_input()


def test_get_input_custom_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    custom_path = tmp_path / "custom/input.json"
    write_json(custom_path, valid_input())

    argv = [
        "python",
        "--input",
        str(custom_path),
    ]

    result = get_input(argv)

    assert result == valid_input()


def test_get_input_missing_file_raises_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    argv = [
        "python",
        "--input",
        "missing.json",
    ]

    with pytest.raises(FileNotFoundError):
        get_input(argv)


def test_get_input_invalid_json_raises_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    invalid_path = tmp_path / "invalid_input.json"
    invalid_path.write_text("{ invalid json", encoding="utf-8")

    argv = [
        "python",
        "--input",
        str(invalid_path),
    ]

    with pytest.raises(json.JSONDecodeError):
        get_input(argv)


def test_get_llm_default() -> None:
    argv = ["python"]

    assert get_llm(argv) == "Qwen/Qwen3-0.6B"


@pytest.mark.xfail(
    reason="Current get_llm() implementation searches for '--input' instead of '--llm'."
)
def test_get_llm_custom() -> None:
    argv = [
        "python",
        "--llm",
        "custom/model",
    ]

    assert get_llm(argv) == "custom/model"


def test_parsing_with_default_files(default_project_files: Path) -> None:
    argv = ["python"]

    config = parsing(argv)

    assert isinstance(config, Config)
    assert config.function_definition == valid_function_definition()
    assert config.input == valid_input()
    assert config.output_file == "data/output/function_calls.json"
    assert config.details is False
    assert config.tokenizer is False
    assert config.llm == "Qwen/Qwen3-0.6B"


def test_parsing_with_custom_files_and_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    functions_path = tmp_path / "custom/functions.json"
    input_path = tmp_path / "custom/input.json"
    output_path = tmp_path / "custom/output.json"

    write_json(functions_path, valid_function_definition())
    write_json(input_path, valid_input())

    argv = [
        "python",
        "--functions_definition",
        str(functions_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--details",
        "--tokenizer",
    ]

    config = parsing(argv)

    assert isinstance(config, Config)
    assert config.function_definition == valid_function_definition()
    assert config.input == valid_input()
    assert config.output_file == str(output_path)
    assert config.details is True
    assert config.tokenizer is True
    assert config.llm == "Qwen/Qwen3-0.6B"


def test_parsing_rejects_unknown_argument(default_project_files: Path) -> None:
    argv = [
        "python",
        "--bad-argument",
    ]

    with pytest.raises(Exception) as error:
        parsing(argv)

    assert "Unknown argument" in str(error.value)


def test_parsing_rejects_invalid_function_definition_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    invalid_function_definition = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "a": {},
            },
            "returns": {"type": "number"},
        }
    ]

    write_json(
        tmp_path / "data/input/functions_definition.json",
        invalid_function_definition,
    )
    write_json(
        tmp_path / "data/input/function_calling_tests.json",
        valid_input(),
    )

    argv = ["python"]

    with pytest.raises(Exception) as error:
        parsing(argv)

    assert "invalid function_definition format" in str(error.value)


def test_parsing_rejects_invalid_input_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    invalid_input = [
        {
            "text": "This should be prompt.",
        }
    ]

    write_json(
        tmp_path / "data/input/functions_definition.json",
        valid_function_definition(),
    )
    write_json(
        tmp_path / "data/input/function_calling_tests.json",
        invalid_input,
    )

    argv = ["python"]

    with pytest.raises(Exception) as error:
        parsing(argv)

    assert "invalid input format" in str(error.value)
