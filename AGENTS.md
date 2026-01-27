# Repository Guidelines

## Project Structure & Module Organization

- This repository is a Python package rooted at the repo top-level (see `__init__.py`), with subpackages in `module_*/`.
- Core modules live in `module_abstraction/`, `module_concretization/`, `module_extraction_by_name/`, `module_naming/`, plus supporting modules like `module_formatter/` and `module_semantic_parallel_splitter/`.
- Common patterns inside a module: `module.py` (DSPy module), `signatures.py` (DSPy signatures), optional `metrics.py`, `optimize.py`, and occasional `demo_test.py` scripts.
- Shared utilities: `config/` (LLM configuration) and `utils/` (helpers).
- Data and artifacts: `datasets/`, `epistack_data/`, `data_models/`. Treat `build/`, `*.egg-info/`, and `venv/` as generated/local.

## Build, Test, and Development Commands

- Create an environment: `python3 -m venv .venv && source .venv/bin/activate`
- Install (editable): `python3 -m pip install -e .`
- Install dev tools (lint/format/test): `python3 -m pip install -e ".[dev]"`
- Smoke-check imports: `python3 test_imports.py`
- Run the demo entrypoint: `python3 main.py`
- Run optimization (calls LLMs; writes JSON): `python3 run_optimization.py`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes.
- Format with Black and lint with Ruff (no repo-specific config): `python3 -m black .` and `python3 -m ruff check .`
- Keep the public API stable: if you add/rename user-facing symbols, update exports in `__init__.py`.

## Testing Guidelines

- Current “tests” are import and demo scripts (`test_imports.py`, `module_*/demo_test.py`). Prefer adding real unit tests under `tests/` using `pytest` (`test_*.py`).
- Run tests with: `python3 -m pytest -q`

## Commit & Pull Request Guidelines

- Commit subjects follow a lightweight Conventional Commits style (e.g. `feat: ...`) mixed with imperative summaries; keep them short and action-oriented.
- PRs should include: what changed, how to run/verify (commands), and any LLM/dataset impact (new env vars, new files under `datasets/`, expected outputs).

## Security & Configuration Tips

- LLM config is loaded from `.env` via `python-dotenv` (see `config/llm.py`). Required vars: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (optional `OPENROUTER_API_BASE`).
- Never commit secrets or `.env` files; treat optimization scripts as potentially costly because they call external LLM APIs.
