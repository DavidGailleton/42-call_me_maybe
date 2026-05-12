from src.classes.Config import Config


def get_output_file(argv: list[str]) -> str:
    for i in range(len(argv) - 1):
        if argv[i + 1] == "--output" and i < len(argv) - 2:
            return argv[i + 2]
    return "data/output/function_calls.json"


def test_args(argv: list[str]) -> None:
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
                        "--function_definition param has multiple"
                        "definition in args"
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


def get_function_definition(
    argv: list[str],
) -> list[dict[str, str | dict[str, str | dict[str, str]]]]:
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
    with open(file_name, "r") as file:
        content = json.load(file)
    return content


def get_input(argv: list[str]) -> list[dict[str, str]]:
    import json

    file_name: str | None = None
    if "--input" in argv:
        try:
            file_name = [
                argv[i + 1] for i in range(len(argv)) if argv[i] == "--input"
            ][0]
        except IndexError:
            pass
    if file_name is None:
        file_name = "data/input/function_calling_tests.json"
    with open(file_name, "r") as file:
        content = json.load(file)
    return content


def parsing(argv: list[str]) -> Config | None:
    fn_def = get_function_definition(argv)
    input = get_input(argv)
    output = get_output_file(argv)
    print(input)

    config = Config(
        function_definition=fn_def,
        input=input,
        output_file=output,
    )
    print(config.function_definition)
    return config
