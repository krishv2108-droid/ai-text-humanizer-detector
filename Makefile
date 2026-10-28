.PHONY: setup test

setup:
	uv sync

test:
	uv run pytest -q
