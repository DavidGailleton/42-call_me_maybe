from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


def get_fn_name(config: Config, prompt: str) -> str:
    model = Small_LLM_Model()
    query = (
        f"Choose the right function name to use based on the descriptio\n"
        f"Prompt: {prompt}\nAvailable functions (name : description) : \n"
    )
    for fn in config.function_definition:
        query += f"{fn['name']} : {fn['description']}\n"
    input_ids = model.encode(query)[0].tolist()
    print(f"encode: {input_ids}")
    logits = model.get_logits_from_input_ids(input_ids)
    res_tokens: list[int] = []
    res: str = ""
    while True:
        res_tokens.append(input_ids[logits.index(sorted(logits)[0])])
        av_answer = [
            x["name"]
            for x in config.function_definition
            if x["name"].startswith(model.decode(res_tokens))
        ]
        if len(av_answer) == 1:
            res = av_answer[0]
            break
    return res


def process_data(config: Config) -> None:
    print(config.input)
    print(get_fn_name(config, config.input[0]["prompt"]))
