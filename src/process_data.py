import json
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

    def unique_preserve_order(self, values: list[str]) -> list[str]:
        """
        Remove duplicate strings while preserving order.

        Args:
            values: Candidate strings.

        Returns:
            Deduplicated candidate strings.
        """
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)

        return result

    def extract_quoted_strings(self, prompt: str) -> list[str]:
        """
        Extract single-quoted and double-quoted strings from a prompt.

        Args:
            prompt: User prompt.

        Returns:
            List of quoted string contents without quotes.
        """
        matches = re.findall(r'"([^"]*)"|\'([^\']*)\'', prompt)
        results: list[str] = []

        for double_quoted, single_quoted in matches:
            value = double_quoted or single_quoted
            if value:
                results.append(value)

        return results

    def extract_numbers(self, prompt: str) -> list[str]:
        """
        Extract integer and decimal numbers from a prompt.

        Args:
            prompt: User prompt.

        Returns:
            List of number strings.
        """
        return re.findall(r"-?\d+(?:\.\d+)?", prompt)

    def extract_after_keyword(
        self,
        prompt: str,
        keyword: str,
    ) -> str | None:
        """
        Extract the first simple word after a keyword.

        Example:
            "replace x with NUMBERS" -> "NUMBERS"

        Args:
            prompt: User prompt.
            keyword: Keyword to search.

        Returns:
            Word after the keyword, or None.
        """
        match = re.search(
            rf"\b{re.escape(keyword)}\b\s+([A-Za-z0-9_\-\*]+)",
            prompt,
            re.IGNORECASE,
        )
        if match is None:
            return None
        return match.group(1)

    def get_parameter_candidates(
        self,
        param_type: str,
        prompt: str,
    ) -> list[str]:

        numbers = self.extract_numbers(prompt)
        lowered = prompt.lower()

        if param_type in {"number", "integer"}:
            return self.unique_preserve_order(numbers)

        if param_type == "boolean":
            return ["true", "false"]

        if param_type != "string":
            return []

        return lowered.split()

    def serialize_candidate_value(
        self,
        raw_value: str,
        param_type: str,
    ) -> str:
        """
        Convert a raw candidate into a valid JSON literal.

        Args:
            raw_value: Raw candidate extracted from the prompt.
            param_type: Expected JSON type.

        Returns:
            Candidate serialized as JSON-compatible text.
        """
        if param_type == "string":
            return json.dumps(raw_value)

        if param_type == "boolean":
            lowered = raw_value.lower()
            if lowered in {"true", "yes", "1"}:
                return "true"
            if lowered in {"false", "no", "0"}:
                return "false"
            return "false"

        if param_type == "number":
            return raw_value

        return json.dumps(raw_value)

    def parse_generated_value(
        self,
        generated_text: str,
        param_type: str,
    ) -> Any:
        """
        Parse a generated JSON literal into a Python value.

        Args:
            generated_text: JSON literal selected by constrained decoding.
            param_type: Expected parameter type.

        Returns:
            Parsed Python value.
        """
        value = json.loads(generated_text)

        if param_type == "number":
            return float(value)

        if param_type == "integer":
            return int(value)

        if param_type == "boolean":
            return bool(value)

        return value

    def get_token_mask_parameters(
        self,
        query: str,
        output_ids: list[int],
        candidate_values: list[str],
    ) -> list[int]:
        """
        Build a token mask for generating one JSON parameter value.

        A token is valid only if appending it keeps the generated continuation
        compatible with at least one candidate value.

        Args:
            query: Prompt prefix before the generated value.
            output_ids: Already generated token ids for the value only.
            candidate_values: Allowed serialized JSON literals.

        Returns:
            Vocabulary mask where 1 means allowed and 0 means forbidden.
        """
        vocab_size = self.vocab_size
        mask = [0] * vocab_size
        query_ids = self.model.encode(query)[0].tolist()

        for candidate in candidate_values:
            full_ids = self.model.encode(query + candidate)[0].tolist()

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

    def generate_parameter_value(
        self,
        query: str,
        candidate_values: list[str],
    ) -> str:
        """
        Generate exactly one serialized JSON value using constrained decoding.

        Args:
            query: Prompt prefix before the value.
            candidate_values: Allowed serialized JSON literal candidates.

        Returns:
            The selected serialized JSON literal.

        Raises:
            ValueError: If no valid token or candidate is available.
        """
        if not candidate_values:
            raise ValueError("No candidate values available")

        candidate_values = self.unique_preserve_order(candidate_values)
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
                raise ValueError("No candidate matches generated tokens")

            if len(full_matches) >= 1:
                return full_matches[0]

    def get_fn_parameters(self, fn_name: str, prompt: str) -> dict[str, Any]:
        """
        Extract all parameters for a selected function.

        Each parameter value is generated independently as a JSON literal using
        constrained decoding. The final dictionary is parsed from those JSON
        literals and returned as Python values.

        Args:
            fn_name: Selected function name.
            prompt: User prompt.

        Returns:
            Dictionary of parameter names to parsed Python values.

        Raises:
            ValueError: If the function or a parameter value cannot be found.
        """
        matching_definitions = [
            definition
            for definition in self.config.function_definition
            if definition["name"] == fn_name
        ]

        if not matching_definitions:
            raise ValueError(f"Function not found: {fn_name}")

        definition = matching_definitions[0]
        parameters = definition["parameters"]

        if not isinstance(parameters, dict):
            raise ValueError(f"Invalid parameters for function: {fn_name}")

        result: dict[str, Any] = {}

        for parameter_index, item in enumerate(parameters.items()):
            parameter, parameter_definition = item

            if not isinstance(parameter_definition, dict):
                raise ValueError(f"Invalid parameter definition: {parameter}")

            param_type = parameter_definition.get("type")

            if not isinstance(param_type, str):
                raise ValueError(f"Missing parameter type: {parameter}")

            raw_candidates = self.get_parameter_candidates(
                param_type=param_type,
                prompt=prompt,
            )

            serialized_candidates = [
                self.serialize_candidate_value(candidate, param_type)
                for candidate in raw_candidates
            ]

            serialized_candidates = self.unique_preserve_order(
                serialized_candidates
            )

            if not serialized_candidates:
                raise ValueError(
                    f"No candidate found for parameter '{parameter}'"
                )

            query = (
                "system\n"
                "You extract one function parameter value from a user request.\n"
                "You must choose the best value for the target parameter.\n"
                "Return exactly one JSON value from the allowed values.\n"
                "Do not return an object.\n"
                "Do not return the parameter name.\n"
                "Do not explain anything.\n\n"
                "Selection rules:\n"
                "- Use the function definition to understand the task.\n"
                "- Use the target parameter name to understand which value is needed.\n"
                "- If several values are possible, choose the one whose role matches "
                "the target parameter.\n"
                "- Prefer exact text from the user prompt unless normalization is needed.\n"
                "- Do not reuse an already selected value unless it is clearly correct.\n\n"
                f"Function definition:\n{definition}\n\n"
                f"Target parameter name:\n{parameter}\n"
                f"Target parameter type:\n{param_type}\n\n"
                f"Already selected parameters:\n{result}\n\n"
                f"Allowed JSON values:\n{serialized_candidates}\n\n"
                "user\n"
                f"{prompt}\n"
                "assistant\n"
            )

            generated_text = self.generate_parameter_value(
                query=query,
                candidate_values=serialized_candidates,
            )

            result[parameter] = self.parse_generated_value(
                generated_text,
                param_type,
            )

        return result


def process_data(config: Config, llm: str = "Qwen/Qwen3-0.6B") -> None:
    solver = PromptSolver(config, llm)
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
