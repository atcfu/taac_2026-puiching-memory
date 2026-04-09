from __future__ import annotations

from ...domain.experiment import ExperimentSpec


def _unique_sorted_ints(values: list[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) > 0})


def _unique_sorted_floats(values: list[float]) -> list[float]:
    return sorted({round(float(value), 6) for value in values if float(value) >= 0.0})


def _round_hidden_dim(value: int) -> int:
    return max(8, ((int(value) + 7) // 8) * 8)


def _hidden_dim_choices(base_hidden_dim: int) -> list[int]:
    return _unique_sorted_ints(
        [
            _round_hidden_dim(base_hidden_dim // 2),
            _round_hidden_dim(base_hidden_dim),
            _round_hidden_dim(int(base_hidden_dim * 3 / 2)),
            _round_hidden_dim(base_hidden_dim * 2),
        ]
    )


def _num_heads_choices(hidden_dim: int, base_num_heads: int) -> list[int]:
    preferred = [1, 2, 4, 8, 16]
    choices = [
        num_heads
        for num_heads in preferred
        if hidden_dim % num_heads == 0 and (hidden_dim // num_heads) % 2 == 0
    ]
    if (
        base_num_heads not in choices
        and base_num_heads > 0
        and hidden_dim % base_num_heads == 0
        and (hidden_dim // base_num_heads) % 2 == 0
    ):
        choices.append(base_num_heads)
    return _unique_sorted_ints(choices) or [1]


def _all_num_heads_choices(hidden_dims: list[int], base_num_heads: int) -> list[int]:
    choices: list[int] = []
    for hidden_dim in hidden_dims:
        choices.extend(_num_heads_choices(hidden_dim, base_num_heads))
    return _unique_sorted_ints(choices)


def _head_hidden_dim_choices(hidden_dim: int, base_head_hidden_dim: int | None) -> list[int]:
    values = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
    if base_head_hidden_dim is not None:
        values.append(int(base_head_hidden_dim))
    return _unique_sorted_ints(values)


def _all_head_hidden_dim_choices(hidden_dims: list[int], base_head_hidden_dim: int | None) -> list[int]:
    choices: list[int] = []
    for hidden_dim in hidden_dims:
        choices.extend(_head_hidden_dim_choices(hidden_dim, base_head_hidden_dim))
    return _unique_sorted_ints(choices)


def _ffn_multiplier_choices(base_ffn_multiplier: float) -> list[float]:
    return _unique_sorted_floats([2.0, 3.0, 4.0, base_ffn_multiplier])


def _dropout_choices(base_dropout: float, *, upper: float) -> list[float]:
    return _unique_sorted_floats(
        [
            0.0,
            min(max(base_dropout - 0.05, 0.0), upper),
            min(max(base_dropout, 0.0), upper),
            min(max(base_dropout + 0.05, 0.0), upper),
            upper,
        ]
    )


def _pairwise_weight_choices(base_pairwise_weight: float) -> list[float]:
    return _unique_sorted_floats(
        [
            0.0,
            min(max(base_pairwise_weight - 0.05, 0.0), 0.5),
            min(max(base_pairwise_weight, 0.0), 0.5),
            min(max(base_pairwise_weight + 0.05, 0.0), 0.5),
            0.25,
        ]
    )


def build_default_search_experiment(base_experiment: ExperimentSpec, trial) -> ExperimentSpec:
    experiment = base_experiment.clone()
    hidden_dim_choices = _hidden_dim_choices(base_experiment.model.hidden_dim)
    num_head_choices = _all_num_heads_choices(hidden_dim_choices, base_experiment.model.num_heads)
    head_hidden_dim_choices = _all_head_hidden_dim_choices(hidden_dim_choices, base_experiment.model.head_hidden_dim)

    hidden_dim = trial.suggest_categorical(
        "model.hidden_dim",
        hidden_dim_choices,
    )
    experiment.model.hidden_dim = hidden_dim
    experiment.model.embedding_dim = hidden_dim
    experiment.model.num_heads = trial.suggest_categorical(
        "model.num_heads",
        num_head_choices,
    )
    if (
        hidden_dim % experiment.model.num_heads != 0
        or (hidden_dim // experiment.model.num_heads) % 2 != 0
    ):
        raise ValueError(
            f"invalid hidden_dim/num_heads combination: hidden_dim={hidden_dim}, num_heads={experiment.model.num_heads}"
        )
    experiment.model.num_layers = trial.suggest_int(
        "model.num_layers",
        max(1, base_experiment.model.num_layers - 1),
        max(1, base_experiment.model.num_layers + 2),
    )
    experiment.model.head_hidden_dim = trial.suggest_categorical(
        "model.head_hidden_dim",
        head_hidden_dim_choices,
    )
    experiment.model.ffn_multiplier = trial.suggest_categorical(
        "model.ffn_multiplier",
        _ffn_multiplier_choices(base_experiment.model.ffn_multiplier),
    )
    experiment.model.dropout = trial.suggest_categorical(
        "model.dropout",
        _dropout_choices(base_experiment.model.dropout, upper=0.3),
    )
    experiment.model.attention_dropout = trial.suggest_categorical(
        "model.attention_dropout",
        _dropout_choices(base_experiment.model.attention_dropout, upper=0.2),
    )

    experiment.train.learning_rate = trial.suggest_float(
        "train.learning_rate",
        1.0e-5,
        5.0e-3,
        log=True,
    )
    experiment.train.weight_decay = trial.suggest_float(
        "train.weight_decay",
        1.0e-6,
        5.0e-2,
        log=True,
    )
    experiment.train.pairwise_weight = trial.suggest_categorical(
        "train.pairwise_weight",
        _pairwise_weight_choices(base_experiment.train.pairwise_weight),
    )
    return experiment


__all__ = ["build_default_search_experiment"]
