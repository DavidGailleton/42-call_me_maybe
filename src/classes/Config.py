from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Application configuration validated with pydantic.

    The configuration contains all data needed to run the function-calling
    pipeline: function definitions, input prompts, output path, display flags,
    tokenizer choice, and LLM model name.
    """

    function_definition: Any
    input: Any
    output_file: str
    details: bool
    tokenizer: bool
    llm: str = Field(default="Qwen/Qwen3-0.6B")

    @field_validator("function_definition")
    @classmethod
    def validate_function_def(cls, value: Any) -> list[dict[str, Any]]:
        """Validate the function definition structure.

        A valid function definition must be a list of dictionaries. Each
        dictionary must contain:

        - name: string
        - description: string
        - parameters: dictionary
        - returns: dictionary

        Each parameter must also contain a string `type` field.

        Args:
            value: Raw function definition data.

        Returns:
            The validated function definition list.

        Raises:
            ValueError: If the function definition format is invalid.
        """
        if not isinstance(value, list):
            raise ValueError("invalid function_definition format")

        for function_definition in value:
            if not isinstance(function_definition, dict):
                raise ValueError("invalid function_definition format")

            name = function_definition.get("name")
            description = function_definition.get("description")
            parameters = function_definition.get("parameters")
            returns = function_definition.get("returns")

            if not isinstance(name, str):
                raise ValueError("invalid function_definition format")

            if not isinstance(description, str):
                raise ValueError("invalid function_definition format")

            if not isinstance(parameters, dict):
                raise ValueError("invalid function_definition format")

            if not isinstance(returns, dict):
                raise ValueError("invalid function_definition format")

            for parameter_name, parameter_definition in parameters.items():
                if not isinstance(parameter_name, str):
                    raise ValueError("invalid function_definition format")

                if not isinstance(parameter_definition, dict):
                    raise ValueError("invalid function_definition format")

                parameter_type = parameter_definition.get("type")

                if not isinstance(parameter_type, str):
                    raise ValueError("invalid function_definition format")

            return_type = returns.get("type")

            if not isinstance(return_type, str):
                raise ValueError("invalid function_definition format")

        return value

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: Any) -> list[dict[str, str]]:
        """Validate the input prompt structure.

        A valid input must be a list of dictionaries. Each dictionary must
        contain a string field named `prompt`.

        Args:
            value: Raw input data.

        Returns:
            The validated input list.

        Raises:
            ValueError: If the input format is invalid.
        """
        if not isinstance(value, list):
            raise ValueError("invalid input format")

        for prompt_data in value:
            if not isinstance(prompt_data, dict):
                raise ValueError("invalid input format")

            prompt = prompt_data.get("prompt")

            if not isinstance(prompt, str):
                raise ValueError("invalid input format")

        return value
