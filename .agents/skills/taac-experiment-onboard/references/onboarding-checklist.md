# Onboarding Checklist

## Source Of Truth

Read these files before making structural decisions:

- `docs/getting-started.md`
- `docs/architecture.md`
- `docs/experiments/index.md`
- `docs/guide/contributing.md`
- `tests/integration/test_experiment_packages.py`
- `tests/support.py` when adding or expanding training-stack tests
- the closest existing package under `config`

If repository interfaces disagree, follow the pattern used by the touched package and tests.
Do not mix unrelated migrations into a package-onboarding task unless the user asks for it.

## Executable Package Path

Add or update these files:

- `config/<name>/__init__.py`
- `config/<name>/model.py`
- `docs/experiments/<name>.md`

Usually update these files too:

- `docs/experiments/index.md`
- `README.md`
- `docs/index.md`
- `zensical.toml`
- `tests/integration/test_experiment_packages.py`

Conditionally update these files:

- `config/<name>/data.py` only when the default data pipeline is not enough
- `config/<name>/utils.py` only when the default loss or optimizer builders are not enough
- `docs/papers/<name>.md` when a long-form paper digest is worth maintaining
- `docs/papers/index.md` when a new paper page is added
- shared modules in `src/taac2026` only for truly reusable framework logic
- search-related files or tests when the package needs custom search behavior

Keep these rules:

- Export `EXPERIMENT` from `config/<name>/__init__.py`.
- Prefer the minimal package shape: package-local `model.py` plus framework default data / loss / optimizer builders unless the experiment really needs overrides.
- Keep package-private modeling and data glue inside `config/<name>`.
- Prefer adapting the architecture to the local batch/runtime contract over preserving upstream file layout.
- Put any new tests under the staged test directories such as `tests/unit/` or `tests/integration/`, not under `tests/` root.
- Record only validation evidence that can be opened in the current workspace.

## Concept-Only Path

Use the concept-only path when the source cannot be integrated cleanly yet.

Usually add or update:

- `docs/ideas/`
- `docs/papers/<name>.md`
- `zensical.toml` when the page is added to nav

Avoid these changes for concept-only work:

- do not add `config/<name>` as an executable package
- do not add the package to README executable tables
- do not add it to `docs/experiments/index.md` executable package or smoke-result tables
- do not imply train/evaluate/search support exists if it does not

## Validation Matrix

Prefer the smallest set of checks that proves the claimed state:

- Executable package scaffold or contract change: `uv run pytest tests/integration/test_experiment_packages.py -q`
- Model robustness change: `uv run pytest tests/integration/test_model_robustness.py -q`
- Shared runtime change: `uv run pytest tests -q`
- Docs or nav change: `uv run --no-project --isolated --with zensical zensical build --clean`
- Smoke training claim: `uv run taac-train --experiment config/<name>`
- Evaluation claim: `uv run taac-evaluate single --experiment config/<name>`
- Search claim: `uv run taac-search --experiment config/<name> --trials <n>`

When validation is skipped, state why and avoid overstating package readiness.
