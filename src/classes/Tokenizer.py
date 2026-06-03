import array
import json
import unicodedata
from pathlib import Path


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {b: chr(c) for b, c in zip(bs, cs)}


def get_pairs(tokens: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()

    for i in range(len(tokens) - 1):
        pairs.add((tokens[i], tokens[i + 1]))

    return pairs


class Tokenizer:
    def __init__(self, tokenizer_path: str) -> None:
        with Path(tokenizer_path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab: dict[str, int] = data["model"]["vocab"]
        self.id_to_token: dict[int, str] = {
            token_id: token for token, token_id in self.vocab.items()
        }

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {
            token: byte for byte, token in self.byte_encoder.items()
        }

        merges = data["model"].get("merges", [])
        self.bpe_ranks: dict[tuple[str, str], int] = {}

        for index, merge in enumerate(merges):
            if isinstance(merge, str):
                left, right = merge.split()
            else:
                left, right = merge
            self.bpe_ranks[(left, right)] = index

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def pre_tokenize(self, text: str) -> list[str]:
        """Approximate pre-tokenizer.

        This is simpler than the real Qwen regex pre-tokenizer.
        """
        if text == "":
            return []

        pieces: list[str] = []
        current = ""

        for char in text:
            if current == "":
                current = char
                continue

            if char.isspace():
                if current:
                    pieces.append(current)
                current = char
            elif current[-1].isspace():
                current += char
                pieces.append(current)
                current = ""
            elif char.isalnum() == current[-1].isalnum():
                current += char
            else:
                pieces.append(current)
                current = char

        if current:
            pieces.append(current)

        return pieces

    def byte_level_encode(self, text: str) -> str:
        return "".join(
            self.byte_encoder[byte] for byte in text.encode("utf-8")
        )

    def apply_bpe(self, text: str) -> list[str]:
        if text in self.vocab:
            return [text]

        tokens = tuple(text)

        while len(tokens) > 1:
            pairs = get_pairs(tokens)

            best_pair: tuple[str, str] | None = None
            best_rank: int | None = None

            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            first, second = best_pair
            new_tokens: list[str] = []
            index = 0

            while index < len(tokens):
                if (
                    index < len(tokens) - 1
                    and tokens[index] == first
                    and tokens[index + 1] == second
                ):
                    new_tokens.append(first + second)
                    index += 2
                else:
                    new_tokens.append(tokens[index])
                    index += 1

            tokens = tuple(new_tokens)

        return list(tokens)

    def encode(self, text: str) -> list[array.ArrayType]:
        normalized = self.normalize(text)
        pieces = self.pre_tokenize(normalized)

        ids: list[int] = []

        for piece in pieces:
            byte_piece = self.byte_level_encode(piece)
            bpe_tokens = self.apply_bpe(byte_piece)

            for token in bpe_tokens:
                token_id = self.vocab.get(token)

                if token_id is None:
                    raise ValueError(f"Unknown token after BPE: {token!r}")

                ids.append(token_id)

        return [array.array("i", ids)]

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.id_to_token[token_id] for token_id in ids)

        byte_array = bytearray()

        for char in text:
            byte = self.byte_decoder.get(char)
            if byte is not None:
                byte_array.append(byte)

        return byte_array.decode("utf-8", errors="replace")
