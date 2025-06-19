import dataclasses
from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import optax


class _GradientTransformationWithSchedules(NamedTuple):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    schedules: dict[str, optax.Schedule]


@flax.struct.dataclass
class TrainState:
    step: int
    params: dict
    opt_state: optax.OptState
    rng: jax.random.PRNGKey

    model_def: nn.Module = flax.struct.field(pytree_node=False)
    tx: optax.GradientTransformation | _GradientTransformationWithSchedules = (
        flax.struct.field(pytree_node=False)
    )


@dataclasses.dataclass
class OptimizerConfig:
    lr: float = 6e-4
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-1
    warmup_steps: int = 700
    grad_clip: float | None = 1.0
    wsd: bool = False


def create_tx(config: OptimizerConfig, max_steps: int, accum_steps: int = 1):
    def mask(params):
        return flax.traverse_util.path_aware_map(
            lambda path, x: "kernel" in path, params
        )

    if config.wsd:
        warmup = config.warmup_steps
        lr = optax.join_schedules(
            (
                optax.warmup_constant_schedule(0, config.lr, warmup),
                optax.cosine_decay_schedule(config.lr, int(0.1 * max_steps)),
            ),
            boundaries=(int(0.9 * max_steps),),
        )
    else:
        lr = optax.warmup_cosine_decay_schedule(
            0.0, config.lr, config.warmup_steps, max_steps
        )
    tx = optax.adamw(
        learning_rate=lr,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        mask=mask,
    )
    if config.grad_clip is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            tx,
        )
    if accum_steps > 1:
        tx = optax.MultiSteps(tx, accum_steps)
    return _GradientTransformationWithSchedules(
        tx.init, tx.update, schedules={"lr": lr}
    )
