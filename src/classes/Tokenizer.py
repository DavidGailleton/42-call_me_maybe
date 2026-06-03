import array
import json


class Tokenizer:
    def __init__(self, vocab_path: str) -> None:
        with open(vocab_path, "r", encoding="utf-8") as file:
            raw_vocab = json.load(file)

        self.id_to_token: dict[int, str] = {
            int(token_id): token for token, token_id in raw_vocab.items()
        }

        self.token_to_id: dict[str, int] = {
            token: token_id for token_id, token in self.id_to_token.items()
        }

        self.tokens_by_length: list[str] = sorted(
            self.token_to_id.keys(),
            key=len,
            reverse=True,
        )

    def encode(self, text: str) -> list[array.ArrayType]:
        result: list[int] = []

        for word in [
            "Ġ" + w if i > 0 else w for i, w in enumerate(text.split(" "))
        ]:
            index = 0
            while index < len(word):
                matched_token: str | None = None

                for token in self.tokens_by_length:
                    if word.startswith(token, index):
                        matched_token = token
                        break

                if matched_token is None:
                    index += 1
                    continue

                result.append(self.token_to_id[matched_token])
                index += len(matched_token)

        return [array.array("i", result)]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.id_to_token[token_id] for token_id in token_ids
        ).replace("Ġ", " ")
