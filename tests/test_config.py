import pytest
from pydantic import ValidationError

from src.classes.Config import Config


def valid_function_definition() -> list[dict]:
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


def test_config_valid_data() -> None:
    config = Config(
        function_definition=valid_function_definition(),
        input=valid_input(),
        output_file="data/output/result.json",
        details=False,
        tokenizer=False,
    )

    assert config.function_definition[0]["name"] == "fn_add_numbers"
    assert config.input[0]["prompt"] == "What is the sum of 2 and 3?"
    assert config.output_file == "data/output/result.json"
    assert config.details is False
    assert config.tokenizer is False
    assert config.llm == "Qwen/Qwen3-0.6B"


def test_config_custom_llm() -> None:
    config = Config(
        function_definition=valid_function_definition(),
        input=valid_input(),
        output_file="out.json",
        details=True,
        tokenizer=True,
        llm="custom/model",
    )

    assert config.llm == "custom/model"


def test_config_missing_function_name() -> None:
    function_definition = [
        {
            "description": "Add two numbers together.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        }
    ]

    with pytest.raises(ValidationError) as error:
        Config(
            function_definition=function_definition,
            input=valid_input(),
            output_file="out.json",
            details=False,
            tokenizer=False,
        )

    assert "invalid function_definition format" in str(error.value)


def test_config_missing_function_description() -> None:
    function_definition = [
        {
            "name": "fn_add_numbers",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        }
    ]

    with pytest.raises(ValidationError) as error:
        Config(
            function_definition=function_definition,
            input=valid_input(),
            output_file="out.json",
            details=False,
            tokenizer=False,
        )

    assert "invalid function_definition format" in str(error.value)


def test_config_missing_parameters() -> None:
    function_definition = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "returns": {"type": "number"},
        }
    ]

    with pytest.raises(ValidationError) as error:
        Config(
            function_definition=function_definition,
            input=valid_input(),
            output_file="out.json",
            details=False,
            tokenizer=False,
        )

    assert "invalid function_definition format" in str(error.value)


def test_config_missing_parameter_type() -> None:
    function_definition = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "a": {},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        }
    ]

    with pytest.raises(ValidationError) as error:
        Config(
            function_definition=function_definition,
            input=valid_input(),
            output_file="out.json",
            details=False,
            tokenizer=False,
        )

    assert "invalid function_definition format" in str(error.value)


def test_config_missing_returns() -> None:
    function_definition = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        }
    ]

    with pytest.raises(ValidationError) as error:
        Config(
            function_definition=function_definition,
            input=valid_input(),
            output_file="out.json",
            details=False,
            tokenizer=False,
        )

    assert "invalid function_definition format" in str(error.value)


def test_config_valid_empty_function_definition() -> None:
    config = Config(
        function_definition=[],
        input=valid_input(),
        output_file="out.json",
        details=False,
        tokenizer=False,
    )

    assert config.function_definition == []


def test_config_valid_empty_input() -> None:
    config = Config(
        function_definition=valid_function_definition(),
        input=[],
        output_file="out.json",
        details=False,
        tokenizer=False,
    )

    assert config.input == []


def test_config_missing_prompt_key() -> None:
    input_data = [
        {"text": "What is the sum of 2 and 3?"},
    ]

    with pytest.raises(ValidationError) as error:
        Config(
            function_definition=valid_function_definition(),
            input=input_data,
            output_file="out.json",
            details=False,
            tokenizer=False,
        )

    assert "invalid input format" in str(error.value)


def test_config_invalid_input_type() -> None:
    with pytest.raises(ValidationError):
        Config(
            function_definition=valid_function_definition(),
            input="not a list",
            output_file="out.json",
            details=False,
            tokenizer=False,
        )


def test_config_invalid_function_definition_type() -> None:
    with pytest.raises(ValidationError):
        Config(
            function_definition="not a list",
            input=valid_input(),
            output_file="out.json",
            details=False,
            tokenizer=False,
        )


def test_config_missing_required_output_file() -> None:
    with pytest.raises(ValidationError):
        Config(
            function_definition=valid_function_definition(),
            input=valid_input(),
            details=False,
            tokenizer=False,
        )


def test_config_missing_required_details() -> None:
    with pytest.raises(ValidationError):
        Config(
            function_definition=valid_function_definition(),
            input=valid_input(),
            output_file="out.json",
            tokenizer=False,
        )


def test_config_missing_required_tokenizer() -> None:
    with pytest.raises(ValidationError):
        Config(
            function_definition=valid_function_definition(),
            input=valid_input(),
            output_file="out.json",
            details=False,
        )
