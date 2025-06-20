import dataclasses
from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import optax


def repo_path():
    from etils import epath

    return epath.Path(__file__).parent


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
        return jax.tree.map(lambda x: x.ndim >= 2, params)

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


def load_pretrained(file):
    from functools import partial

    import jax.numpy as jnp
    import safetensors.numpy
    from etils import epath

    import src.llm as llm
    import src.transformer as transformer

    file = epath.Path(file)
    loaded_params = safetensors.numpy.load(file.read_bytes())
    loaded_params = {
        k.replace("h.", "transformer.blocks_"): v for k, v in loaded_params.items()
    }
    loaded_params = {
        k: v for k, v in loaded_params.items() if not k.endswith(".attn.bias")
    }
    loaded_params = flax.traverse_util.unflatten_dict(loaded_params, sep=".")

    config = transformer.GPTConfig(vocab_size=50257)
    model_def = llm.LLM(
        transformer.Transformer(config),
        width=config.n_embd,
        vocab_size=config.vocab_size,
        max_seq_len=config.block_size,
        dtype=jnp.bfloat16,
    )

    jax.eval_shape(
        partial(model_def.apply, train=False),
        {"params": loaded_params},
        jnp.ones((1, 32), dtype=jnp.int32),
    )

    return TrainState(
        step=0,
        params=loaded_params,
        rng=jax.random.PRNGKey(0),
        model_def=model_def,
        tx=None,
        opt_state=None,
    )
