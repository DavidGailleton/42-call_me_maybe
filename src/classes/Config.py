from pydantic import BaseModel, field_validator


class Config(BaseModel):
    function_definition: list[dict[str, str | dict[str, str | dict[str, str]]]]
    input: list[dict[str, str]]
    output_file: str

    @field_validator("function_definition")
    def validate_function_def(cls, v: list):
        try:
            for fn in v:
                fn["name"]
                fn["description"]
                fn["parameters"]
                for param in fn["parameters"]:
                    fn["parameters"][param]["type"]
                fn["returns"]
        except KeyError:
            raise ValueError("invalid function_definition format")

    @field_validator("input")
    def validate_input(cls, v: list):
        try:
            for prompt in v:
                prompt["prompt"]
        except KeyError:
            raise ValueError("invalid input format")
