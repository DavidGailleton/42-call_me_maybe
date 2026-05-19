import json
from os import read
import re
from typing import Any

from pydantic import config
from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


class PromptSolver:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = Small_LLM_Model()

    class FunctionNotFound(Exception):
        pass

    def get_token_mask_fn_name(
        self,
        input_ids: list[int],
        query: str,
        fn_names: list[str],
    ) -> list[int]:
        """
        Return a mask over the vocabulary for the next valid token.
        input_ids contains the whole prompt + generated continuation.
        """
        query_ids = self.model.encode(query)[0].tolist()
        generated_ids = input_ids[len(query_ids) :]

        with open(
            self.model.get_path_to_vocab_file(), "r", encoding="utf-8"
        ) as f:
            vocab = json.load(f)

        vocab_size = len(vocab)
        mask = [0] * vocab_size

        for name in fn_names:
            full_ids = self.model.encode(query + name)[0].tolist()

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

    def get_next_token_id(
        self, logits: list[float], token_mask: list[int] | None = None
    ) -> int:
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
        if token_mask is None:
            token_mask = [1] * len(logits)
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

    def get_fn_name(self, prompt: str) -> str:
        query = (
            f"You must choose exactly one function name from the list below.\n"
            f"Output only the function name, with no explanation and no extra text.\n\n"
            f"User request:\n{prompt}\n\nAvailable functions:\n{self.config.function_definition}\n"
        )
        query += "\nAnswer="
        fn_name: list[str] = [
            fn["name"] for fn in self.config.function_definition
        ]
        input_ids = self.model.encode(query)[0].tolist()
        while True:
            input_ids.append(
                self.get_next_token_id(
                    self.model.get_logits_from_input_ids(input_ids),
                    self.get_token_mask_fn_name(input_ids, query, fn_name),
                )
            )
            av_answer: list = [
                name
                for name in fn_name
                if name.startswith(
                    self.model.decode(input_ids).removeprefix(query).strip()
                )
                or self.model.decode(input_ids).removeprefix(query).strip()
                == ""
            ]
            if len(av_answer) == 1:
                return av_answer[0]

    def get_token_mask_parameters(
        self,
        query: str,
        input_ids: list[int],
        candidate_values: list[str] | None,
    ) -> list[int]:
        """
        Return a vocabulary mask for the next token when generating one parameter value.

        A token is allowed iff appending it keeps the generated continuation compatible
        with at least one candidate value.
        """
        with open(
            self.model.get_path_to_vocab_file(), "r", encoding="utf-8"
        ) as file:
            vocab = json.load(file)

        vocab_size = len(vocab)
        if candidate_values is None:
            return [1] * vocab_size
        mask = [0] * vocab_size

        query_ids = self.model.encode(query)[0].tolist()
        generated_ids = input_ids[len(query_ids) :]

        for value in candidate_values:
            full_ids = self.model.encode(query + value)[0].tolist()

            if full_ids[: len(query_ids)] != query_ids:
                continue

            continuation_ids = full_ids[len(query_ids) :]

            if len(generated_ids) > len(continuation_ids):
                continue
            if continuation_ids[: len(generated_ids)] != generated_ids:
                continue

            if len(generated_ids) < len(continuation_ids):
                next_token_id = continuation_ids[len(generated_ids)]
                mask[next_token_id] = 1

        return mask

    def get_fn_parameters(self, fn_name: str, prompt: str) -> dict:
        definition = [
            definition
            for definition in self.config.function_definition
            if definition["name"] == fn_name
        ][0]
        res: dict[str, Any] = {}
        for parameter in definition["parameters"]:
            query = (
                "You must extract exactly one parameters value from the prompt.\n"
                "Output only the parameters, with no explanation and no extra text.\n\n"
                f"function definition:\n{definition}\n\n"
                f"Parameter to return:\n{parameter}\n\n"
                f"Actual answer parameters: {res}\n\n"
                f"Base prompt: {prompt}\n\n"
                "Answer="
            )
            input_ids = self.model.encode(query)[0].tolist()
            try:
                while True:
                    if definition["parameters"][parameter]["type"] == "number":
                        candidate_values = re.findall(
                            r"-?\d+(?:\.\d+)?", prompt
                        )
                    elif (
                        definition["parameters"][parameter]["type"]
                        == "boolean"
                    ):
                        candidate_values = ["true", "false"]
                    else:
                        candidate_values = None

                    input_ids.append(
                        self.get_next_token_id(
                            self.model.get_logits_from_input_ids(input_ids),
                            self.get_token_mask_parameters(
                                query, input_ids, candidate_values
                            ),
                        )
                    )
                    if (
                        definition["parameters"][parameter]["type"]
                        == "boolean"
                    ):
                        raise ValueError
                    if definition["parameters"][parameter][
                        "type"
                    ] == "string" and self.model.decode(input_ids).endswith(
                        "\n"
                    ):
                        raise ValueError
            except ValueError:
                if definition["parameters"][parameter]["type"] == "number":
                    res[parameter] = float(
                        self.model.decode(input_ids)
                        .removeprefix(query)
                        .strip()
                    )
                else:
                    res[parameter] = float(
                        self.model.decode(input_ids)
                        .removeprefix(query)
                        .strip()
                    )
        return res


def process_data(config: Config) -> None:
    solver = PromptSolver(config)
    output: list[dict[str, str | dict[str, Any]]] = []
    for prompt in config.input:
        output.append(
            {
                "prompt": prompt["prompt"],
                "name": solver.get_fn_name(prompt["prompt"]),
            }
        )
    for o in output:
        o["parameters"] = solver.get_fn_parameters(o["name"], o["prompt"])
        print(o)
    print(output)
