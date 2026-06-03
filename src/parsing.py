from src.classes.Config import Config


def get_output_file(argv: list[str]) -> str:
    for i in range(len(argv) - 1):
        if argv[i + 1] == "--output" and i < len(argv) - 2:
            return argv[i + 2]
    return "data/output/function_calls.json"


def test_args(args: list[str]) -> None:
    i: int = 1
    param_found: dict = {
        "--functions_definition": 0,
        "--input": 0,
        "--output": 0,
        "--details": 0,
        "--llm": 0,
        "--tokenizer": 0,
    }
    while i < len(args):
        match args[i]:
            case "--functions_definition":
                if param_found["--functions_definition"] == 1:
                    raise Exception(
                        "--functions_definition param has multiple"
                        "definition in args"
                    )
                i += 2
                param_found["--functions_definition"] = 1
            case "--input":
                if param_found["--input"] == 1:
                    raise Exception(
                        "--input param has multiple definition in args"
                    )
                i += 2
                param_found["--input"] = 1
            case "--output":
                if param_found["--output"] == 1:
                    raise Exception(
                        "--output param has multiple definition in args"
                    )
                i += 2
                param_found["--output"] = 1
            case "--llm":
                if param_found["--llm"] == 1:
                    raise Exception(
                        "--llm param has multiple definition in args"
                    )
                i += 2
                param_found["--llm"] = 1
            case "--details":
                if param_found["--details"] == 1:
                    raise Exception(
                        "--details param has multiple definition in args"
                    )
                i += 1
                param_found["--details"] = 1
            case "--tokenizer":
                if param_found["--tokenizer"] == 1:
                    raise Exception(
                        "--tokenizer param has multiple definition in args"
                    )
                i += 1
                param_found["--tokenizer"] = 1
            case _:
                raise Exception(f"Unknown argument: {args[i]}")


def get_function_definition(
    argv: list[str],
) -> list[dict[str, str | dict[str, str | dict[str, str]]]]:
    import json

    file_name: str | None = None
    if "--functions_definition" in argv:
        try:
            file_name = [
                argv[i + 1]
                for i in range(len(argv))
                if argv[i] == "--functions_definition"
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


def get_llm(argv: list[str]) -> str:
    llm: str = "Qwen/Qwen3-0.6B"
    if "--llm" in argv:
        try:
            llm = [
                argv[i + 1] for i in range(len(argv)) if argv[i] == "--input"
            ][0]
        except IndexError:
            pass
    return llm


def parsing(argv: list[str]) -> Config | None:
    test_args(argv)

    fn_def = get_function_definition(argv)
    input = get_input(argv)
    output = get_output_file(argv)
    llm = get_llm(argv)

    config = Config(
        function_definition=fn_def,
        input=input,
        output_file=output,
        details="--details" in argv,
        tokenizer="--tokenizer" in argv,
        llm=llm,
    )
    return config
