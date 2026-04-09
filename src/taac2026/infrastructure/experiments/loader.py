from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from ...domain.experiment import ExperimentSpec
from ..io.files import stable_hash64


def _load_from_path(path: Path):
    module_name = f"taac2026_dynamic_{stable_hash64(str(path))}"
    init_file = path / "__init__.py" if path.is_dir() else path
    spec = importlib.util.spec_from_file_location(module_name, init_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _derive_import_name(path: Path) -> str | None:
    resolved = path.resolve()
    if resolved.is_dir():
        module_root = resolved
    elif resolved.name == "__init__.py":
        module_root = resolved.parent
    else:
        module_root = resolved.with_suffix("")

    for entry in sys.path:
        try:
            base = Path(entry or ".").resolve()
            relative = module_root.relative_to(base)
        except (OSError, ValueError):
            continue
        if not relative.parts:
            continue
        if not all(part.isidentifier() for part in relative.parts):
            continue
        return ".".join(relative.parts)
    return None


def load_experiment_package(experiment_path: str | Path) -> ExperimentSpec:
    path = Path(experiment_path)
    if path.exists():
        import_name = _derive_import_name(path)
        if import_name is not None:
            module = importlib.import_module(import_name)
        else:
            module = _load_from_path(path)
    else:
        module = importlib.import_module(str(experiment_path))
    experiment = getattr(module, "EXPERIMENT", None)
    if experiment is None:
        raise AttributeError(f"Experiment package {experiment_path} does not define EXPERIMENT")
    return experiment.clone()


__all__ = ["load_experiment_package"]
