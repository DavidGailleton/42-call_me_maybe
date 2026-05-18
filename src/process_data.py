import json
from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


def get_token_mask_fn_name(
    input_ids: list[int],
    query: str,
    model: Small_LLM_Model,
    fn_names: list[str],
) -> list[int]:
    """
    Return a mask over the vocabulary for the next valid token.
    input_ids contains the whole prompt + generated continuation.
    """
    query_ids = model.encode(query)[0].tolist()
    generated_ids = input_ids[len(query_ids) :]

    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    mask = [0] * vocab_size

    for name in fn_names:
        full_ids = model.encode(query + name)[0].tolist()

        if full_ids[: len(query_ids)] != query_ids:
            continue

        continuation = full_ids[len(query_ids) :]

        if len(generated_ids) > len(continuation):
            continue
        if continuation[: len(generated_ids)] != generated_ids:
            continue

        if len(generated_ids) < len(continuation):
            next_token_id = continuation[len(generated_ids)]
            mask[next_token_id] = 1

    return mask


def get_next_token_id(logits: list[float], token_mask: list[int]) -> int:
    """
    Return the highest-scoring allowed token id.

    Args:
        logits: Scores for each token in the vocabulary.
        token_mask: 1 for allowed tokens, 0 for forbidden tokens.

    Returns:
        The selected token id.

    Raises:
        ValueError: If sizes differ or no token is allowed.
    """
    if len(logits) > len(token_mask):
        logits = logits[0 : len(token_mask)]

    best_token_id: int | None = None
    best_score = float("-inf")

    for token_id, score in enumerate(logits):
        if token_mask[token_id] == 1 and score > best_score:
            best_score = score
            best_token_id = token_id

    if best_token_id is None:
        raise ValueError("No valid token available in token_mask")

    return best_token_id


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
        input_ids.append(
            get_next_token_id(
                model.get_logits_from_input_ids(input_ids),
                get_token_mask_fn_name(input_ids, query, model, fn_name),
            )
        )
        av_answer: list = [
            name
            for name in fn_name
            if name.startswith(
                model.decode(input_ids).removeprefix(query).strip()
            )
            or model.decode(input_ids).removeprefix(query).strip() == ""
        ]
        if len(av_answer) == 1:
            return av_answer[0]


def process_data(config: Config) -> None:
    print(get_fn_name(config, config.input[0]["prompt"]))
