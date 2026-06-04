from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Sequence, cast

from llm_sdk import Small_LLM_Model
from src.classes.Config import Config
from src.classes.Tokenizer import Tokenizer


class PromptSolver:
    """Solve natural-language prompts into structured function calls.

    The solver uses the LLM to select a function and extract parameters. Function
    name generation is constrained so that the model can only produce one of the
    declared function names.
    """

    class FunctionNotFound(Exception):
        """Raised when no valid function can be selected."""

    def __init__(self, config: Config) -> None:
        """Initialize the prompt solver.

        Args:
            config: Runtime configuration containing input data and model setup.
        """
        self.config = config
        self.function_definitions = cast(
            list[dict[str, Any]],
            self.config.function_definition,
        )

        self.model: Any = Small_LLM_Model(model_name=self.config.llm)
        self.tokenizer = Tokenizer(self.model.get_path_to_tokenizer_file())

        with open(
            self.model.get_path_to_vocab_file(),
            "r",
            encoding="utf-8",
        ) as file:
            self.vocab: Any = json.load(file)

        self.vocab_size = len(self.vocab)

    def _normalise_encoded(self, encoded: Any) -> list[int]:
        """Convert tokenizer output into a flat list of token IDs.

        Args:
            encoded: Token IDs returned by either the SDK or custom tokenizer.

        Returns:
            A flat list of integer token IDs.

        Raises:
            TypeError: If the encoded value cannot be converted.
        """
        if hasattr(encoded, "tolist"):
            raw = encoded.tolist()
        else:
            raw = encoded

        if (
            isinstance(raw, list)
            and len(raw) == 1
            and hasattr(raw[0], "tolist")
        ):
            raw = raw[0].tolist()

        if (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], list)
        ):
            raw = raw[0]

        if not isinstance(raw, list):
            raise TypeError("encoded token value must be list-like")

        return [int(token_id) for token_id in raw]

    def encode_text(self, text: str) -> list[int]:
        """Encode text with the selected tokenizer.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        if self.config.tokenizer:
            return self._normalise_encoded(self.tokenizer.encode(text))
        return self._normalise_encoded(self.model.encode(text))

    def decode_ids(self, token_ids: list[int]) -> str:
        """Decode token IDs into text.

        Args:
            token_ids: Token IDs to decode.

        Returns:
            Decoded text.
        """
        if self.config.tokenizer:
            return self.tokenizer.decode(token_ids)

        decoded = self.model.decode(token_ids)
        if isinstance(decoded, str):
            return decoded
        if isinstance(decoded, list):
            return "".join(decoded)
        return str(decoded)

    def _normalise_logits(self, logits: Any) -> list[float]:
        """Convert model logits into a flat list of floats.

        Args:
            logits: Raw logits returned by the LLM SDK.

        Returns:
            Flat list of scores for the next token.
        """
        raw = logits

        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        if hasattr(raw, "tolist"):
            raw = raw.tolist()

        while (
            isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list)
        ):
            raw = raw[-1]

        if not isinstance(raw, list):
            raise TypeError("logits must be list-like")

        return [float(score) for score in raw]

    def get_logits(self, token_ids: list[int]) -> list[float]:
        """Return next-token logits for the given token IDs.

        Args:
            token_ids: Input token IDs.

        Returns:
            List of logits.
        """
        raw_logits = self.model.get_logits_from_input_ids(token_ids)
        return self._normalise_logits(raw_logits)

    def get_next_token_id(
        self,
        logits: Sequence[float],
        token_mask: Sequence[int] | None = None,
    ) -> int:
        """Return the highest-scoring allowed token ID.

        Args:
            logits: Score for each token in the vocabulary.
            token_mask: Optional mask where 1 means allowed and 0 means blocked.

        Returns:
            Selected token ID.

        Raises:
            ValueError: If no valid token can be selected.
        """
        if token_mask is None:
            token_mask = [1] * len(logits)

        current_logits = list(logits)

        if len(current_logits) > len(token_mask):
            current_logits = current_logits[: len(token_mask)]

        if len(current_logits) != len(token_mask):
            raise ValueError("logits and token_mask must have the same size")

        best_token_id: int | None = None
        best_score = float("-inf")

        for token_id, score in enumerate(current_logits):
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
        """Build a token mask allowing only valid function-name continuations.

        Args:
            query: Prompt prefix before the function name.
            output_ids: Already generated function-name token IDs.
            fn_names: Allowed function names.

        Returns:
            Token mask over the vocabulary.
        """
        query_ids = self.encode_text(query)
        mask = [0] * self.vocab_size

        for name in fn_names:
            full_ids = self.encode_text(query + name)

            if full_ids[: len(query_ids)] != query_ids:
                continue

            continuation = full_ids[len(query_ids) :]

            if len(output_ids) > len(continuation):
                continue
            if continuation[: len(output_ids)] != output_ids:
                continue

            if len(output_ids) < len(continuation):
                next_token_id = continuation[len(output_ids)]
                if 0 <= next_token_id < self.vocab_size:
                    mask[next_token_id] = 1

        return mask

    def get_fn_name(self, prompt: str) -> str:
        """Select the best function name for a prompt.

        Args:
            prompt: Natural-language user request.

        Returns:
            Selected function name.

        Raises:
            FunctionNotFound: If no function can be selected.
        """
        fn_def_formatted = {
            str(fn["name"]): str(fn["description"])
            for fn in self.function_definitions
        }

        query = (
            "<|im_start|>system\n"
            "You must choose exactly one function name from the list below.\n"
            "Output only the function name, "
            "with no explanation and no extra text.<|im_end|>\n"
            f"Available functions:\n{fn_def_formatted}\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        fn_names = [str(fn["name"]) for fn in self.function_definitions]

        input_ids = self.encode_text(query)
        output_ids: list[int] = []
        query_ids = self.encode_text(query)

        while True:
            full_ids = input_ids + output_ids
            logits = self.get_logits(full_ids)
            token_mask = self.get_token_mask_fn_name(
                query,
                output_ids,
                fn_names,
            )

            next_token_id = self.get_next_token_id(logits, token_mask)
            output_ids.append(next_token_id)

            available_answers: list[str] = []
            for name in fn_names:
                full_name_ids = self.encode_text(query + name)
                continuation = full_name_ids[len(query_ids) :]
                if continuation[: len(output_ids)] == output_ids:
                    available_answers.append(name)

            if len(available_answers) == 1:
                full_name_ids = self.encode_text(query + available_answers[0])
                continuation = full_name_ids[len(query_ids) :]
                if continuation == output_ids:
                    return available_answers[0]

    def is_valid_json(self, token_ids: list[int]) -> bool:
        """Check whether decoded token IDs contain balanced JSON braces.

        Args:
            token_ids: Generated output token IDs.

        Returns:
            True if the number of opening and closing braces is balanced.
        """
        output = self.decode_ids(token_ids)
        nb_open_bracket = 0
        nb_close_bracket = 0

        for char in output:
            if char == "{":
                nb_open_bracket += 1
            elif char == "}":
                nb_close_bracket += 1

        return nb_open_bracket == nb_close_bracket

    def escape_invalid_json_backslashes(self, text: str) -> str:
        """Escape invalid JSON backslashes in generated text.

        Args:
            text: Raw generated JSON-like text.

        Returns:
            Text with invalid backslashes escaped.
        """
        return re.sub(
            r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
            r"\\\\",
            text,
        )

    def convert_format(
        self,
        output: dict[str, Any],
        param_definition: dict[str, str],
    ) -> str:
        """Convert extracted parameters to the expected schema types.

        Args:
            output: Raw generated parameter dictionary.
            param_definition: Mapping of parameter names to expected types.

        Returns:
            JSON-compatible string representation of converted parameters.
        """
        result: dict[str, Any] = {}

        for param, param_type in param_definition.items():
            try:
                if param_type == "number":
                    result[param] = float(output[param])
                elif param_type == "integer":
                    result[param] = int(output[param])
                else:
                    result[param] = output[param]
            except Exception:
                result[param] = "error"

        return json.dumps(result, ensure_ascii=False)

    def get_formated_output(
        self,
        fn_name: str,
        definition: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Generate the final structured output for one prompt.

        Args:
            fn_name: Selected function name.
            definition: Function definition dictionary.
            prompt: Original user prompt.

        Returns:
            JSON-compatible dictionary containing prompt, name, and parameters.
        """
        parameters: dict[str, str] = {
            str(key): str(value["type"])
            for key, value in definition["parameters"].items()
        }

        query = f"""<|im_start|>system
You are a function calling argument extractor.

Your task is to extract the arguments for the selected function from the user request.<|im_end|>
<|im_start|>user
User request:
{json.dumps(prompt)}

Selected function:
{json.dumps(fn_name)}

Required parameters:
{json.dumps(parameters)}

Return only a JSON object containing exactly the required parameters.
Do not include explanations.
Do not include markdown.

Example format:
    {{
    \"prompt\": {json.dumps(prompt)},
    \"name\": \"{fn_name}\",
    \"parameters\": {{
    }}
}},<|im_end|>
<|im_start|>assistant"""

        base_output = f"""
{{
    \"prompt\": {json.dumps(prompt)},
    \"name\": \"{fn_name}\",
    \"parameters\": {{"""

        output_ids = self.encode_text(base_output)
        input_ids = self.encode_text(query)

        while True:
            logits = self.get_logits(input_ids + output_ids)
            output_ids.append(self.get_next_token_id(logits, None))

            if self.config.details:
                os.system("cls" if os.name == "nt" else "clear")
                decoded = self.decode_ids(output_ids)
                print(f"""Prompt: {prompt}

Output: {decoded}""")

            if len(output_ids) > len(input_ids):
                raise ValueError("Output not found")

            if self.is_valid_json(output_ids):
                decoded_output = self.escape_invalid_json_backslashes(
                    self.decode_ids(output_ids)
                )
                parsed_output = json.loads(decoded_output)

                result = {
                    "prompt": prompt,
                    "name": fn_name,
                    "parameters": json.loads(
                        self.convert_format(
                            cast(dict[str, Any], parsed_output)["parameters"],
                            parameters,
                        )
                    ),
                }

                return result


def process_data(config: Config) -> None:
    """Process all prompts and write function-call results to a JSON file.

    Args:
        config: Runtime configuration.
    """
    solver = PromptSolver(config)
    prompts = cast(list[dict[str, str]], config.input)
    function_definitions = cast(
        list[dict[str, Any]], config.function_definition
    )

    output: list[dict[str, Any]] = []

    for prompt_data in prompts:
        prompt = prompt_data["prompt"]

        try:
            fn_name = solver.get_fn_name(prompt)
            definition = [
                d for d in function_definitions if d["name"] == fn_name
            ][0]
            output.append(
                solver.get_formated_output(fn_name, definition, prompt)
            )
        except Exception as err:
            print(err)
            output.append({"prompt": prompt, "name": "error"})

    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(config.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=4, ensure_ascii=False)
