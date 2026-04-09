from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import DataConfig, ModelConfig, SearchConfig, TrainConfig


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    build_data_pipeline: Callable[..., Any]
    build_model_component: Callable[..., Any]
    build_loss_stack: Callable[..., Any]
    build_optimizer_component: Callable[..., Any]
    switches: dict[str, bool] = field(default_factory=dict)
    search: SearchConfig = field(default_factory=SearchConfig)
    build_search_experiment: Callable[["ExperimentSpec", Any], "ExperimentSpec"] | None = None

    def clone(self) -> "ExperimentSpec":
        return ExperimentSpec(
            name=self.name,
            data=deepcopy(self.data),
            model=deepcopy(self.model),
            train=deepcopy(self.train),
            build_data_pipeline=self.build_data_pipeline,
            build_model_component=self.build_model_component,
            build_loss_stack=self.build_loss_stack,
            build_optimizer_component=self.build_optimizer_component,
            switches=deepcopy(self.switches),
            search=deepcopy(self.search),
            build_search_experiment=self.build_search_experiment,
        )

    def derive(self, **updates: Any) -> "ExperimentSpec":
        experiment = self.clone()
        for key, value in updates.items():
            if not hasattr(experiment, key):
                raise AttributeError(f"ExperimentSpec has no field '{key}'")
            setattr(experiment, key, value)
        return experiment


__all__ = ["ExperimentSpec"]
