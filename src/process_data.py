from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


def get_token_mask(
    input_ids: list[int],
    query: str,
    model: Small_LLM_Model,
    fn_name: list[str],
) -> list[int]:
    i = 0
    res = []
    print(f"\n\nfn_name: {fn_name}\n\n")
    answer = model.decode(input_ids).removeprefix(query)
    print(answer)
    if answer != "":
        fn_name = [
            fn.removeprefix(answer) for fn in fn_name if fn.startswith(answer)
        ]
    print(f"\n\nfn_name: {fn_name}\n\n")
    while len(input_ids) > i:
        if (
            sorted(
                [
                    1 if fn.startswith(model.decode([input_ids[i]])) else 0
                    for fn in fn_name
                ],
                reverse=True,
            )[0]
            == 1
        ):
            res.append(1)
        else:
            res.append(0)
        i += 1
    return input_ids


def get_fn_name(config: Config, prompt: str) -> str:
    model = Small_LLM_Model()
    query = (
        f"You must choose exactly one function name from the list below.\n"
        f"Output only the function name, with no explanation and no extra text.\n\n"
        f"User request:\n{prompt}\n\nAvailable functions:\n{config.function_definition}\n"
    )
    query += "\nAnswer="
    fn_name: list[str] = [fn["name"] for fn in config.function_definition]
    input_ids = model.encode(query)[0].tolist()
    while True:
        masked_token = get_token_mask(input_ids, query, model, fn_name)
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = [
            x if masked_token[logits.index(x)] != 0 else float("-inf")
            for x in logits[0 : len(input_ids)]
        ]
        input_ids.append(
            input_ids[
                masked_logits.index(
                    sorted(
                        masked_logits[0 : len(input_ids)],
                        reverse=True,
                    )[0]
                )
            ]
        )
        av_answer: list = [
            name
            for name in fn_name
            if name.startswith(
                model.decode(input_ids).removeprefix(query).strip()
            )
            or model.decode(input_ids).removeprefix(query).strip() == ""
        ]
        print(f"{model.decode(input_ids)};")
        if len(av_answer) == 1:
            return av_answer[0]


def process_data(config: Config) -> None:
    print(get_fn_name(config, config.input[0]["prompt"]))
