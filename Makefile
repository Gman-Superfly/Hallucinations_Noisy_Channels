.PHONY: lint figures qa

lint:
	python -m ruff check .

figures:
	python scripts/generate_figures.py

qa: lint figures

