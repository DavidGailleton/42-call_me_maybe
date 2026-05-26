import json
import re
from typing import Any

from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


class PromptSolver:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as file:
            self.vocab = json.load(file)
        self.vocab_size = len(self.vocab)

    class FunctionNotFound(Exception):
        pass

    def get_next_token_id(
        self, logits: list[float], token_mask: list[int] | None = None
    ) -> int:
        """
        Return the highest-scoring allowed token id.
        """
        if token_mask is None:
            token_mask = [1] * len(logits)

        if len(logits) > len(token_mask):
            logits = logits[: len(token_mask)]

        if len(logits) != len(token_mask):
            raise ValueError("logits and token_mask must have the same size")

        best_token_id: int | None = None
        best_score = float("-inf")

        for token_id, score in enumerate(logits):
            if token_mask[token_id] == 1 and score > best_score:
                best_score = score
                best_token_id = token_id

        if best_token_id is None:
            raise ValueError("No valid token available in token_mask")

        return best_token_id

    def get_token_mask_fn_name(
        self,
        query: str,
        output_ids: list[int],
        fn_names: list[str],
    ) -> list[int]:
        """
        Return a mask over the vocabulary for the next valid token.
        output_ids contains only the generated continuation.
        """
        query_ids = self.model.encode(query)[0].tolist()
        vocab_size = self.vocab_size
        mask = [0] * vocab_size

        for name in fn_names:
            full_ids = self.model.encode(query + name)[0].tolist()

            if full_ids[: len(query_ids)] != query_ids:
                continue

            continuation = full_ids[len(query_ids) :]

            if len(output_ids) > len(continuation):
                continue
            if continuation[: len(output_ids)] != output_ids:
                continue

            if len(output_ids) < len(continuation):
                next_token_id = continuation[len(output_ids)]
                if 0 <= next_token_id < vocab_size:
                    mask[next_token_id] = 1

        return mask

    def get_fn_name(self, prompt: str) -> str:
        fn_def_formated = {
            fn["name"]: fn["description"]
            for fn in self.config.function_definition
        }
        query = (
            "<|im_start|>system\n"
            "You must choose exactly one function name from the list below.\n"
            "Output only the function name, with no explanation and no extra text.<|im_end|>\n"
            f"Available functions:\n{fn_def_formated}\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        fn_names: list[str] = [
            fn["name"] for fn in self.config.function_definition
        ]

        input_ids = self.model.encode(query)[0].tolist()
        output_ids: list[int] = []

        query_ids = self.model.encode(query)[0].tolist()

        while True:
            full_ids = input_ids + output_ids
            logits = self.model.get_logits_from_input_ids(full_ids)
            token_mask = self.get_token_mask_fn_name(
                query, output_ids, fn_names
            )

            next_token_id = self.get_next_token_id(logits, token_mask)
            output_ids.append(next_token_id)

            av_answer: list[str] = []
            for name in fn_names:
                full_name_ids = self.model.encode(query + name)[0].tolist()
                continuation = full_name_ids[len(query_ids) :]
                if continuation[: len(output_ids)] == output_ids:
                    av_answer.append(name)

            if len(av_answer) == 1:
                full_name_ids = self.model.encode(query + av_answer[0])[
                    0
                ].tolist()
                continuation = full_name_ids[len(query_ids) :]
                if continuation == output_ids:
                    return av_answer[0]

    def get_token_mask_parameters(
        self,
        query: str,
        output_ids: list[int],
        candidate_values: list[str],
    ) -> list[int]:
        vocab_size = self.vocab_size
        mask = [0] * vocab_size
        query_ids = self.model.encode(query)[0].tolist()

        for value in candidate_values:
            full_ids = self.model.encode(query + value)[0].tolist()

            if full_ids[: len(query_ids)] != query_ids:
                continue

            continuation_ids = full_ids[len(query_ids) :]

            if len(output_ids) > len(continuation_ids):
                continue
            if continuation_ids[: len(output_ids)] != output_ids:
                continue

            if len(output_ids) < len(continuation_ids):
                next_token_id = continuation_ids[len(output_ids)]
                if 0 <= next_token_id < vocab_size:
                    mask[next_token_id] = 1

        return mask

    def get_fn_parameters(self, fn_name: str, prompt: str) -> dict[str, Any]:
        definition = [
            definition
            for definition in self.config.function_definition
            if definition["name"] == fn_name
        ][0]

        res: dict[str, Any] = {}

        for parameter, parameter_definition in definition[
            "parameters"
        ].items():
            param_type = parameter_definition["type"]

            query = (
                "system\n"
                "You must extract exactly one parameter value from the prompt.\n\n"
                "Output only the parameter value, with no explanation and no extra text.\n\n"
                f"function definition:\n{definition}\n\n"
                f"Parameter to return:\n{parameter}\n\n"
                f"Already selected parameters: {res}\n"
                "user\n"
                f"{prompt}\n"
                "assistant\n"
            )

            if param_type == "number":
                candidate_values = re.findall(r"-?\d+(?:\.\d+)?", prompt)

            elif param_type == "boolean":
                lowered = prompt.lower()
                candidate_values = []
                if "true" in lowered or "yes" in lowered:
                    candidate_values.append("true")
                if "false" in lowered or "no" in lowered:
                    candidate_values.append("false")
                if not candidate_values:
                    candidate_values = ["true", "false"]

            elif param_type == "string":
                quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', prompt)
                strings = []

                for a, b in quoted:
                    value = a or b
                    if value:
                        strings.append(value)

                if not strings:
                    strings = re.findall(r"[A-Za-z0-9_\-]+", prompt)

                candidate_values = strings

            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            if not candidate_values:
                raise ValueError(
                    f"No candidate found for parameter '{parameter}'"
                )

            query_ids = self.model.encode(query)[0].tolist()
            output_ids: list[int] = []

            while True:
                logits = self.model.get_logits_from_input_ids(
                    query_ids + output_ids
                )

                token_mask = self.get_token_mask_parameters(
                    query,
                    output_ids,
                    candidate_values,
                )

                next_token_id = self.get_next_token_id(logits, token_mask)
                output_ids.append(next_token_id)

                matching_candidates: list[str] = []
                full_matches: list[str] = []

                for candidate in candidate_values:
                    full_candidate_ids = self.model.encode(query + candidate)[
                        0
                    ].tolist()
                    continuation = full_candidate_ids[len(query_ids) :]

                    if continuation[: len(output_ids)] == output_ids:
                        matching_candidates.append(candidate)

                        if continuation == output_ids:
                            full_matches.append(candidate)

                if not matching_candidates:
                    raise ValueError(
                        f"No valid continuation for parameter '{parameter}'"
                    )

                if len(full_matches) == 1:
                    chosen_value = full_matches[0]

                    if param_type == "number":
                        res[parameter] = float(chosen_value)
                    elif param_type == "boolean":
                        res[parameter] = chosen_value == "true"
                    else:
                        res[parameter] = chosen_value

                    break

        return res


def process_data(config: Config) -> None:
    solver = PromptSolver(config)
    output: list[dict[str, str | dict[str, Any]]] = []
    for i, prompt in enumerate(config.input):
        try:
            output.append(
                {
                    "prompt": prompt["prompt"],
                    "name": solver.get_fn_name(prompt["prompt"]),
                }
            )
            output[i]["parameters"] = solver.get_fn_parameters(
                output[i]["name"], output[i]["prompt"]
            )
        except Exception as err:
            print(f"{err}\n\n")
    print()
    with open(config.output_file, "w", encoding="utf-8") as o_file:
        json.dump(output, o_file, indent=4, ensure_ascii=False)
