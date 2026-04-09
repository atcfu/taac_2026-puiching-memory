# Rules And Hotspots

## Approved Patterns

- Declare required packages in `pyproject.toml`.
- Run project tasks with `uv run ...`.
- Use `uv sync --locked` for a full local environment.
- Use `uv run --with package ...` for temporary tools.
- Use `importlib` only when dynamic loading is the intended product behavior.

## Disallowed Patterns

- `try: import ...` followed by `except ModuleNotFoundError`
- `try: import ...` followed by `except ImportError` to silently disable behavior
- wrappers like `_require_x()` whose only job is to hide undeclared dependencies
- docs or comments that recommend `pip install`, `python script.py`, or bare `pytest`

## Preferred Replacements

- Replace optional-import wrappers with direct imports after declaring the dependency.
- If a feature should stay separate, move it behind a dedicated module or command and run it with an environment explicitly provisioned by `uv`.
- Update user-facing commands to `uv run ...`.
- Keep `pyproject.toml` and `uv.lock` synchronized when dependencies change.

## Current Cleanup Candidates

These existing files still use patterns that conflict with this skill and are good follow-up cleanup targets:

- `src/taac2026/search.py`
- `src/taac2026/training_artifacts.py`
- `src/taac2026/application/training/artifacts.py`

These files use dynamic loading as product behavior and should not be confused with banned try-import fallbacks:

- `src/taac2026/experiment_loader.py`
- `src/taac2026/infrastructure/experiments/loader.py`
