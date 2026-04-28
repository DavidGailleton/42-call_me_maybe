from sys import argv
from pydantic import BaseModel


class Config(BaseModel):
    function_definition: dict
    intput: dict
    output_file: str


def get_output_file() -> str:
    for i in range(len(argv) - 1):
        if argv[i + 1] == "--output" and i < len(argv) - 2:
            return argv[i + 2]
    return "data/output/function_calls.json"


def test_args() -> None:
    i: int = 1
    param_found: dict = {
        "--function_definition": 0,
        "--input": 0,
        "--output": 0,
    }
    while i < len(argv):
        match argv[i]:
            case "--function_definition":
                if param_found["--function_definition"] == 1:
                    raise Exception(
                        "--function_definition param has multiple definition in args"
                    )
                i += 1
                param_found["--function_definition"] = 1
            case "--input":
                if param_found["--input"] == 1:
                    raise Exception(
                        "--input param has multiple definition in args"
                    )
                i += 1
                param_found["--input"] = 1
            case "--output":
                if param_found["--output"] == 1:
                    raise Exception(
                        "--output param has multiple definition in args"
                    )
                i += 1
                param_found["--output"] = 1
            case _:
                raise Exception(f"Unknown argument: {argv[i]}")


def get_function_definition() -> dict:
    import json

    file_name: str | None = None
    if "--function_definition" in argv:
        try:
            file_name = [
                argv[i + 1]
                for i in range(len(argv))
                if argv[i] == "--function_definition"
            ][0]
        except IndexError:
            pass
    if file_name is None:
        file_name = "data/input/functions_definition.json"


def parsing() -> str | None:
    test_args()
