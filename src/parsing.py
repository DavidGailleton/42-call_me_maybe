from __future__ import annotations
from pathlib import Path

import json
from typing import Any, cast

from src.classes.Config import Config

FunctionDefinition = dict[str, Any]
PromptInput = dict[str, str]


def _get_option_value(argv: list[str], option: str) -> str | None:
    """Return the value following a command-line option.

    Args:
        argv: Full command-line argument list.
        option: Option name to search for.

    Returns:
        The option value if present, otherwise None.
    """
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def get_output_file(argv: list[str]) -> str:
    """Return the output file path from arguments or the default path.

    Args:
        argv: Full command-line argument list.

    Returns:
        Output file path.
    """
    output: str | None = _get_option_value(argv, "--output")
    if output is None:
        output = "data/output/function_calls.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    return output


def test_args(args: list[str]) -> None:
    """Validate supported command-line arguments.

    Args:
        args: Full command-line argument list.

    Raises:
        ValueError: If an argument is unknown, duplicated, or missing a value.
    """
    value_options = {
        "--functions_definition",
        "--input",
        "--output",
        "--llm",
    }
    flag_options = {
        "--details",
        "--tokenizer",
    }
    found: dict[str, int] = {
        "--functions_definition": 0,
        "--input": 0,
        "--output": 0,
        "--details": 0,
        "--llm": 0,
        "--tokenizer": 0,
    }

    index = 1
    while index < len(args):
        arg = args[index]

        if arg in value_options:
            if found[arg] == 1:
                raise ValueError(f"{arg} parameter is defined multiple times")
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"{arg} requires a value")
            found[arg] = 1
            index += 2
        elif arg in flag_options:
            if found[arg] == 1:
                raise ValueError(f"{arg} parameter is defined multiple times")
            found[arg] = 1
            index += 1
        else:
            raise ValueError(f"Unknown argument: {arg}")


def _load_json_file(file_name: str) -> Any:
    """Load and parse a JSON file.

    Args:
        file_name: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def get_function_definition(argv: list[str]) -> list[FunctionDefinition]:
    """Load the function definition file.

    Args:
        argv: Full command-line argument list.

    Returns:
        List of function definition dictionaries.
    """
    file_name = _get_option_value(argv, "--functions_definition")
    if file_name is None:
        file_name = "data/input/functions_definition.json"

    try:
        content = _load_json_file(file_name)
    except FileNotFoundError:
        raise Exception("Function definition file not found")
    except json.JSONDecodeError:
        raise Exception("Invalid function definition file format")

    if not isinstance(content, list):
        raise ValueError("functions_definition file must contain a JSON array")

    return cast(list[FunctionDefinition], content)


def get_input(argv: list[str]) -> list[PromptInput]:
    """Load the prompt input file.

    Args:
        argv: Full command-line argument list.

    Returns:
        List of prompt dictionaries.
    """
    file_name = _get_option_value(argv, "--input")
    if file_name is None:
        file_name = "data/input/function_calling_tests.json"

    try:
        content = _load_json_file(file_name)
    except FileNotFoundError:
        raise Exception("Input file not found")
    except json.JSONDecodeError:
        raise Exception("Invalid input file format")

    if not isinstance(content, list):
        raise ValueError("input file must contain a JSON array")

    return cast(list[PromptInput], content)


def get_llm(argv: list[str]) -> str:
    """Return the selected LLM model name.

    Args:
        argv: Full command-line argument list.

    Returns:
        Model name.
    """
    llm = _get_option_value(argv, "--llm")
    if llm is None:
        return "Qwen/Qwen3-0.6B"
    return llm


def parsing(argv: list[str]) -> Config | None:
    """Parse command-line arguments and build the application configuration.

    Args:
        argv: Full command-line argument list.

    Returns:
        A Config instance, or None if parsing should stop.
    """
    test_args(argv)

    fn_def = get_function_definition(argv)
    input_data = get_input(argv)
    output = get_output_file(argv)
    llm = get_llm(argv)

    return Config(
        function_definition=fn_def,
        input=input_data,
        output_file=output,
        details="--details" in argv,
        tokenizer="--tokenizer" in argv,
        llm=llm,
    )
