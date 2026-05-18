from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


def test(config: Config) -> None:
    model = Small_LLM_Model()
    import json

    tools_block = "\n".join(
        json.dumps(fn) for fn in config.function_definition
    )

    query = (
        "<|im_start|>system\n"
        "You are a function calling assistant.\n"
        "Choose the single best function for the user request.\n"
        "Output only the function name.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "greet 'hello'\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for fn in config.function_definition:
        query += f"\n{fn['name']} : \"{fn['description']}\"\n"
    input_ids = model.encode(query)[0].tolist()
    res_tokens: list[int] = []
    while True:
        logits = model.get_logits_from_input_ids(input_ids + res_tokens)
        res_tokens.append(
            input_ids[
                logits.index(
                    sorted(
                        logits[0 : len(input_ids + res_tokens)], reverse=True
                    )[0]
                )
            ]
        )
        print(logits[0 : len(input_ids + res_tokens)])
        print(f"{model.decode(res_tokens)};")


def main() -> None:
    from src import parsing, process_data
    import sys

    print("PARSING : \n")
    config = parsing(sys.argv)
    if config is None:
        return
    print("\n\nPROCESS_DATA : \n")
    process_data(config)


if __name__ == "__main__":
    main()
