SRCS= main.py

install:
	uv sync

run: install
	uv run python3 -m src

debug:
	uv run python3 -m src

clean:
	rm -rf */**/__pycache__ */__pycache__ __pycache__ */.mypy_cache .mypy_cache .venv dist build */**/*.egg-info */*.egg-info *.egg-info test.txt

fclean: clean
	rm mazegen-1.0.0-py3-none-any.whl 

lint: install
	uv run flake8 . --exclude=.venv
	uv run python3 -m mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
lint-strict: install
	uv run flake8 . --exclude=.venv
	uv run  python3 -m mypy --strict .


.PHONY: build install run debug clean fclean lint lint-strict run_test
