*This project has been created as part of the 42 curriculum by dgaillet.*

# call me maybe

## Description

`call me maybe` is an introduction to **function calling with Large Language Models**.

The goal of the project is to transform a natural-language prompt into a structured function call.  
Instead of answering the user directly, the program must identify:

- the function that should be called;
- the arguments required by that function;
- the correct argument types according to the function definition file.

Example:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The project uses a small LLM, by default `Qwen/Qwen3-0.6B`, through the provided `llm_sdk` package.

A key part of the project is **constrained decoding**: instead of trusting the model to freely generate valid structured output, the decoder restricts the possible next tokens so that the model can only generate valid choices.

---

## Instructions

### Requirements

The project is written in **Python 3.10+**.

Dependencies are managed with `uv`.

Required packages include:

- `pydantic`
- `numpy`
- `llm_sdk` provided with the subject

The project must be run from the repository root.

---

### Installation

```bash
uv sync
```

Or using the Makefile:

```bash
make install
```

---

### Running the program

Default execution:

```bash
uv run python -m src
```

Or:

```bash
make run
```

By default, the program reads:

```txt
data/input/functions_definition.json
data/input/function_calling_tests.json
```

and writes the result to:

```txt
data/output/function_calls.json
```

---

### Custom input and output files

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

---

### Optional flags

#### Display generation details

```bash
uv run python -m src --details
```

or:

```bash
make run_details
```

#### Use the custom tokenizer implementation

```bash
uv run python -m src --tokenizer
```

---

### Debug mode

```bash
make debug
```

---

### Linting

```bash
make lint
```

This runs:

```bash
flake8 .
mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

---

### Cleaning temporary files

```bash
make clean
```

---

## Input format

### Function definitions

The function definition file must contain a JSON array of available functions.

Example:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```

Each function must contain:

- `name`
- `description`
- `parameters`
- `returns`

---

### Prompt input file

The prompt file must contain a JSON array of objects.

Example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet Shrek"
  }
]
```

Each object must contain a `prompt` field.

---

## Output format

The program creates a JSON array containing one object per prompt.

Example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet Shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "Shrek"
    }
  }
]
```

Each output object contains:

- `prompt`: the original prompt;
- `name`: the selected function name;
- `parameters`: the extracted arguments.

---

## Algorithm explanation

The project is split into two main steps:

1. **Function selection**
2. **Parameter extraction and formatting**

---

### 1. Function selection

The program first asks the model to choose one function name from the list of available functions.

The available function names and their descriptions are inserted into the prompt.  
The LLM receives the user request and must select the most appropriate function.

However, the model is not allowed to freely generate arbitrary text.

Instead, the function-name generation is performed with constrained decoding.

---

### Constrained decoding approach

Language models generate text token by token.

At each generation step:

1. The current prompt and generated output are encoded into token IDs.
2. The model returns logits for all possible next tokens.
3. The decoder builds a token mask.
4. Invalid tokens are blocked.
5. The highest-scoring valid token is selected.

For function names, the valid outputs are exactly the function names present in `functions_definition.json`.

For example, if the available functions are:

```txt
fn_add_numbers
fn_greet
fn_reverse_string
```

then the decoder only allows token sequences that can still become one of those names.

If the current generated prefix is:

```txt
fn_
```

then the next valid tokens are only tokens that continue at least one available function name.

The generation stops when only one complete function name matches the generated token sequence.

This guarantees that the selected function name is always one of the declared functions.

---

### Function-name token mask

The method responsible for this is:

```python
get_token_mask_fn_name()
```

It works by:

1. Encoding the base query.
2. Encoding each possible function name appended to the query.
3. Extracting the continuation tokens corresponding to the function name.
4. Comparing the already generated output with each possible continuation.
5. Allowing only the next token that keeps at least one valid function name possible.

Then:

```python
get_next_token_id()
```

selects the highest-scoring allowed token.

---

### 2. Parameter extraction

After the function name is selected, the program extracts the required parameters.

The selected function definition is used to know:

- the parameter names;
- the expected parameter types;
- the required structure.

The model is prompted with:

- the original user request;
- the selected function name;
- the parameter schema.

The produced result is then parsed and converted to the expected types.

For example:

```json
{
  "a": "2",
  "b": "3"
}
```

can be converted to:

```json
{
  "a": 2.0,
  "b": 3.0
}
```

depending on the schema.

The final output file itself is written using Python's `json.dump`, which guarantees that the final file is valid JSON.

---

## Design decisions

### Pydantic validation

The project uses `pydantic` for validating configuration data.

The main configuration class is:

```python
Config
```

It validates:

- the function definitions;
- the input prompts;
- the output file path;
- optional flags;
- the selected LLM model.

This helps detect malformed input early.

---

### Separate parsing and processing

The project separates command-line parsing from data processing:

```txt
src/parsing.py
src/process_data.py
```

This makes the code easier to test and maintain.

`parsing.py` is responsible for:

- reading command-line arguments;
- loading input JSON files;
- creating a `Config` object.

`process_data.py` is responsible for:

- initializing the LLM;
- selecting functions;
- extracting parameters;
- writing the output file.

---

### Graceful error handling

The main entry point catches unexpected exceptions:

```python
try:
    ...
except Exception as err:
    print(err)
```

During processing, if one prompt fails, the program continues with the next prompt instead of stopping completely.

This avoids losing all results because of one malformed or ambiguous prompt.

---

### Custom tokenizer

The project includes a custom tokenizer implementation in:

```txt
src/classes/Tokenizer.py
```

It implements:

- byte-to-unicode mapping;
- byte-level encoding;
- BPE pair merging;
- token decoding.

The `--tokenizer` flag allows testing this tokenizer instead of directly using the SDK tokenizer.

This is useful for understanding how tokenization interacts with constrained decoding.

---

## Performance analysis

### Accuracy

The function-name selection step is constrained to the declared function names, which improves reliability compared to free generation.

The model still decides which function is most likely based on the prompt and the function descriptions.

Expected accuracy depends on:

- clarity of the function descriptions;
- ambiguity of the user prompts;
- similarity between available functions;
- quality of the selected LLM.

For simple function sets such as arithmetic, greeting, and string manipulation, the expected accuracy is high.

---

### Reliability

The final output file is written with `json.dump`, so the generated file is always valid JSON if the program reaches the writing step.

The function name is constrained to known function names.

Parameter extraction is schema-aware and attempts to convert values to the expected types.

---

### Speed

The program processes prompts sequentially.

For each prompt, it performs:

1. constrained generation for the function name;
2. generation for parameters;
3. JSON parsing and formatting.

On standard hardware, the expected runtime is acceptable for small test files.

The most expensive part is calling the LLM repeatedly during token-by-token decoding.

---

## Challenges faced

### Small model reliability

Small models are more likely to produce malformed or unexpected output.

To reduce this risk, the function-name selection uses constrained decoding instead of relying only on prompting.

---

### Tokenization

A major challenge is that model output is generated as tokens, not as normal characters.

A valid text prefix may correspond to different token sequences depending on the tokenizer.

To solve this, the implementation encodes the query and each candidate function name, then compares token continuations directly.

---

### JSON generation

LLMs can easily produce invalid JSON when unconstrained.

The project avoids writing raw model output directly to the final file.  
Instead, the final result is converted into Python dictionaries and serialized with `json.dump`.

---

### Type conversion

Model output may contain values as strings even when numbers are expected.

The implementation converts parameters according to the function definition:

- `number` → `float`
- `integer` → `int`
- other types are kept as provided

---

## Testing strategy

The project was tested with several categories of prompts.

### Basic function selection

Examples:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet John"
  },
  {
    "prompt": "Reverse the string hello"
  }
]
```

These tests verify that the correct function is selected.

---

### Argument extraction

Examples:

```json
[
  {
    "prompt": "What is the sum of 265 and 345?"
  },
  {
    "prompt": "Greet Shrek"
  }
]
```

These tests verify that parameters are extracted with the correct names and types.

---

### Edge cases

Useful edge cases include:

```json
[
  {
    "prompt": "Reverse the empty string"
  },
  {
    "prompt": "Add -5 and 12.5"
  },
  {
    "prompt": "Greet a person named Jean-Luc"
  },
  {
    "prompt": "Reverse the string containing quotes: \"hello\""
  }
]
```

These tests check:

- empty values;
- negative numbers;
- decimals;
- punctuation;
- special characters.

---

### Invalid files

The parser should also be tested with:

- missing input files;
- invalid JSON;
- missing `prompt` fields;
- missing function fields;
- invalid parameter schemas.

---

## Example usage

### Example input

`data/input/functions_definition.json`

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```

`data/input/function_calling_tests.json`

```json
[
  {
    "prompt": "What is the sum of 40 and 2?"
  },
  {
    "prompt": "Greet Alice"
  }
]
```

Run:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Possible output:

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 40.0,
      "b": 2.0
    }
  },
  {
    "prompt": "Greet Alice",
    "name": "fn_greet",
    "parameters": {
      "name": "Alice"
    }
  }
]
```

---

## Project structure

```txt
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── data
│   └── input
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk
│   └── ...
└── src
    ├── __init__.py
    ├── __main__.py
    ├── parsing.py
    ├── process_data.py
    └── classes
        ├── Config.py
        └── Tokenizer.py
```

---

## Resources

### Documentation and references

- Python documentation:  
  <https://docs.python.org/3/>

- Python `json` module:  
  <https://docs.python.org/3/library/json.html>

- Pydantic documentation:  
  <https://docs.pydantic.dev/>

- `uv` documentation:  
  <https://docs.astral.sh/uv/>

- Mypy documentation:  
  <https://mypy.readthedocs.io/>

- Flake8 documentation:  
  <https://flake8.pycqa.org/>

- Hugging Face tokenizer documentation, for general tokenizer concepts:  
  <https://huggingface.co/docs/tokenizers/>

---

## AI usage

AI tools were used as learning and assistance tools during the project.

They were used for:

- explaining constrained decoding concepts;
- helping structure the README;
- suggesting possible edge cases for testing;
- generate some unit tests;
- reviewing wording and documentation clarity;
- explaining tokenizer-related concepts.
- help for some code structure (Tokenizer).
- help for increase prompt reliability.

AI was not used as a replacement for understanding the implementation.  
All generated suggestions were reviewed, adapted, and tested before being included in the project.

