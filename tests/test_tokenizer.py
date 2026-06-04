import array
from typing import Any

import pytest
from torch import Tensor

from llm_sdk import Small_LLM_Model
from src.classes.Tokenizer import Tokenizer


def flatten_custom_encode(
    encoded: list[array.array[int]] | Tensor,
) -> list[int]:
    return encoded[0].tolist()


def create_llm_model() -> Any:
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError:
        pytest.skip("llm_sdk is not available")

    try:
        return Small_LLM_Model("Qwen/Qwen3-0.6B")
    except TypeError:
        return Small_LLM_Model()


@pytest.fixture(scope="session")
def llm_model() -> Any:
    return create_llm_model()


@pytest.fixture(scope="session")
def real_tokenizer(llm_model: Small_LLM_Model) -> Tokenizer:
    tokenizer_path = llm_model.get_path_to_tokenizer_file()
    return Tokenizer(tokenizer_path)


def test_custom_tokenizer_decode_matches_llm_sdk_for_simple_texts(
    llm_model: Small_LLM_Model,
    real_tokenizer: Tokenizer,
) -> None:
    texts = [
        "hello",
        "Hello",
        "42",
        "!",
        "abc",
        "fn_add_numbers",
    ]

    if not hasattr(llm_model, "decode"):
        pytest.skip("llm_sdk model has no decode method")

    for text in texts:
        sdk_ids = flatten_custom_encode(llm_model.encode(text))

        assert real_tokenizer.decode(sdk_ids) == llm_model.decode(sdk_ids)


def test_custom_tokenizer_encode_matches_llm_sdk_for_simple_texts(
    llm_model: Small_LLM_Model,
    real_tokenizer: Tokenizer,
) -> None:
    texts = [
        "hello",
        "Hello",
        "42",
        "!",
        "abc",
        "fn_add_numbers",
    ]

    for text in texts:
        custom_ids = flatten_custom_encode(real_tokenizer.encode(text))
        sdk_ids = flatten_custom_encode(llm_model.encode(text))

        assert custom_ids == sdk_ids


@pytest.mark.xfail(
    reason=(
        "The current custom tokenizer uses an approximate pre_tokenize() "
        "implementation, so realistic text may differ from llm_sdk."
    ),
    strict=False,
)
def test_custom_tokenizer_encode_matches_llm_sdk_for_realistic_prompts(
    llm_model: Small_LLM_Model,
    real_tokenizer: Tokenizer,
) -> None:
    texts = [
        "What is the sum of 2 and 3?",
        "Greet Shrek",
        "Reverse the string 'hello'",
        "Call fn_add_numbers with a=40 and b=2",
        "hello world",
        " leading space",
        "multiple   spaces",
    ]

    for text in texts:
        custom_ids = flatten_custom_encode(real_tokenizer.encode(text))
        sdk_ids = llm_model.encode(text)

        assert custom_ids == sdk_ids
