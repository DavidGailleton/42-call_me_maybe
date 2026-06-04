from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Runtime configuration for the function-calling program.

    The class stores validated input data, output options, and model settings.
    Validation is performed at runtime with pydantic because data comes from
    JSON files and command-line arguments.
    """

    function_definition: Any = Field(default_factory=list)
    input: Any = Field(default_factory=list)
    output_file: str = "data/output/function_calls.json"
    details: bool = False
    tokenizer: bool = False
    llm: str = Field(default="Qwen/Qwen3-0.6B")

    @field_validator("function_definition")
    @classmethod
    def validate_function_def(cls, value: Any) -> list[dict[str, Any]]:
        """Validate the function definition JSON structure.

        Args:
            value: Raw function definition data loaded from JSON.

        Returns:
            The validated list of function definitions.

        Raises:
            ValueError: If the function definition format is invalid.
        """
        if not isinstance(value, list):
            raise ValueError("function_definition must be a list")

        for fn in value:
            if not isinstance(fn, dict):
                raise ValueError("each function definition must be an object")

            name = fn.get("name")
            description = fn.get("description")
            parameters = fn.get("parameters")
            returns = fn.get("returns")

            if not isinstance(name, str):
                raise ValueError("function name must be a string")
            if not isinstance(description, str):
                raise ValueError("function description must be a string")
            if not isinstance(parameters, dict):
                raise ValueError("function parameters must be an object")
            if not isinstance(returns, dict):
                raise ValueError("function returns must be an object")

            for param_name, param_def in parameters.items():
                if not isinstance(param_name, str):
                    raise ValueError("parameter names must be strings")
                if not isinstance(param_def, dict):
                    raise ValueError("parameter definition must be an object")
                if not isinstance(param_def.get("type"), str):
                    raise ValueError("parameter type must be a string")

            if not isinstance(returns.get("type"), str):
                raise ValueError("return type must be a string")

        return value

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: Any) -> list[dict[str, str]]:
        """Validate the prompt input JSON structure.

        Args:
            value: Raw prompt data loaded from JSON.

        Returns:
            The validated list of prompt dictionaries.

        Raises:
            ValueError: If the input format is invalid.
        """
        if not isinstance(value, list):
            raise ValueError("input must be a list")

        for prompt in value:
            if not isinstance(prompt, dict):
                raise ValueError("each input item must be an object")
            if not isinstance(prompt.get("prompt"), str):
                raise ValueError(
                    "each input item must contain a prompt string"
                )

        return value
