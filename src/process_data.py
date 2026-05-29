import json
from os import wait
import re
from typing import Any

from transformers import model_addition_debugger_context

from llm_sdk import Small_LLM_Model
from src.classes.Config import Config


class PromptSolver:
    def __init__(self, config: Config, llm: str) -> None:
        self.config = config
        self.model = Small_LLM_Model(model_name=llm)
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

    def is_valid_json(self, token_ids: list[int]) -> bool:
        output = self.model.decode(token_ids)
        nb_open_bracket = 0
        nb_close_bracket = 0
        for char in output:
            if char == "{":
                nb_open_bracket += 1
            elif char == "}":
                nb_close_bracket += 1
        return nb_open_bracket == nb_close_bracket

    def escape_invalid_json_backslashes(self, text: str) -> str:
        return re.sub(
            r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
            r"\\\\",
            text,
        )

    def get_formated_output(
        self, fn_name: str, definition: dict, prompt: str
    ) -> dict:
        parameters: dict = {
            k: v["type"] for k, v in definition["parameters"].items()
        }

        query = f"""You are a function calling argument extractor.

                Your task is to extract the arguments for the selected function from the user request.

                User request:
                \"{prompt}\"

                Selected function:
                \"{fn_name}\"

                Function description:
                Calculate the square root of a number.

                Required parameters:
                {parameters}

                Return only a JSON object containing exactly the required parameters.
                Do not include explanations.
                Do not include markdown.
                

                number is type float should include a dot

                Example format:
                    {{
                    \"prompt\": {json.dumps(prompt)},
                    \"name\": \"{fn_name}\",
                    \"parameters\": {{
                        
                    }}
                }},
                """

        output_ids = list(self.model.encode(f"""
            {{
                \"prompt\": {json.dumps(prompt)},
                \"name\": \"{fn_name}\",""")[0])

        input_ids = list(self.model.encode(query)[0])

        while True:
            logits = self.model.get_logits_from_input_ids(
                input_ids + output_ids
            )

            output_ids.append(self.get_next_token_id(logits, None))

            import os

            os.system("cls" if os.name == "nt" else "clear")
            decoded = self.model.decode(output_ids)
            print(decoded)
            if self.is_valid_json(output_ids):
                return json.loads(
                    self.escape_invalid_json_backslashes(
                        self.model.decode(output_ids)
                    )
                )


def process_data(config: Config, llm: str = "Qwen/Qwen3-0.6B") -> None:
    solver = PromptSolver(config, llm)
    output: list[dict[str, str | dict[str, Any]]] = []
    for prompt_d in config.input:
        # try:
        prompt = prompt_d["prompt"]
        fn_name = solver.get_fn_name(prompt)
        definition = [
            d for d in config.function_definition if d["name"] == fn_name
        ][0]
        output.append(solver.get_formated_output(fn_name, definition, prompt))
    # except Exception as err:
    # print(f"{err}\n\n")
    print()
    with open(config.output_file, "w", encoding="utf-8") as o_file:
        json.dump(output, o_file, indent=4, ensure_ascii=False)
