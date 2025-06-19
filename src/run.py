import functools
import os

import flax
import flax.struct
import flax.traverse_util
import flax_orbax
import jax
import jax.numpy as jnp
import jax.sharding as sharding
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm
import tyro
from absl import logging
from etils import epath

import src.dataset
import src.llm
import src.transformer
import src.utils
import wandb


def main(
    *,
    model: src.transformer.GPTConfig,
    optimizer: src.utils.OptimizerConfig,
    batch_size: int = 512,
    per_device_batch_size: int = 16,  # Max batch size per device
    seq_len: int = 1024,
    num_tokens: int,
    data_dir: str | None = None,
    seed: int = 42,
    save_dir: str | None = None,
    save_interval: int | None = None,
    save_all: bool = False,
):
    model_config = model
    optimizer_config = optimizer

    if data_dir is None:
        data_dir = epath.Path(os.environ["TFDS_DATA_DIR"]) / "fineweb100B"

    jax.distributed.initialize()

    logging.set_verbosity(logging.INFO)
    logging.info(
        f"This is process {jax.process_index()} / {jax.process_count()} ({jax.local_device_count()} local devices)"
    )

    mesh = sharding.Mesh(jax.devices(), axis_names="batch")
    _rs = replicated_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec())
    _ds = dp_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("batch"))

    micro_batch_size = min(batch_size, per_device_batch_size * jax.device_count())
    grad_accum_steps = batch_size // micro_batch_size
    if grad_accum_steps > 1:
        logging.info(
            f"Using {grad_accum_steps} gradient accumulation steps for {batch_size} batch size (w/ {batch_size} micro-batch size)"
        )
    num_steps = num_tokens // (micro_batch_size * seq_len)

    model = src.llm.LLM(
        transformer=src.transformer.Transformer(model_config),
        vocab_size=model_config.vocab_size,
        width=model_config.n_embd,
        max_seq_len=seq_len,
        dtype=jnp.bfloat16,
    )

    tx = flax_orbax.wrap(src.utils.create_tx)(
        optimizer_config, num_steps, grad_accum_steps
    )

    @functools.partial(jax.jit, out_shardings=_rs)
    def init_state():
        key = jax.random.PRNGKey(seed)
        print(
            model.tabulate(
                jax.random.PRNGKey(0),
                jax.numpy.ones((1, 1), dtype=jax.numpy.int32),
                train=False,
            )
        )
        params = model.init(
            key, jax.numpy.ones((1, 1), dtype=jax.numpy.int32), train=False
        )["params"]
        opt_state = tx.init(params)
        return src.utils.TrainState(
            step=jnp.zeros((), dtype=jnp.int32),
            params=params,
            opt_state=opt_state,
            rng=key,
            model_def=model,
            tx=tx,
        )

    @functools.partial(jax.jit, in_shardings=(_rs, _ds), out_shardings=_rs)
    def train_step(state: src.utils.TrainState, tokens: jax.Array):
        def loss_fn(params):
            logits, intermediates = state.model_def.apply(
                {"params": params},
                tokens[:, :-1],
                train=True,
                return_intermediates=True,
                mutable=["intermediates"],
            )
            loss = (
                optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])
                .astype(jnp.float32)
                .mean()
            )
            accuracy = (logits.argmax(axis=-1) == tokens[:, 1:]).mean()
            return loss, {
                "loss": loss,
                "accuracy": accuracy,
            }

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, new_opt_state = state.tx.update(
            updates=grads, state=state.opt_state, params=state.params
        )
        info["grad_norm"] = optax.global_norm(grads)
        info["update_norm"] = optax.global_norm(updates)
        info["param_norm"] = optax.global_norm(state.params)
        info["lr"] = state.tx.schedules["lr"](state.step // grad_accum_steps)

        new_params = optax.apply_updates(state.params, updates)
        state = state.replace(
            params=new_params, opt_state=new_opt_state, step=state.step + 1
        )

        return state, info

    state = init_state()

    dataset = src.dataset.ShardedDataset(
        directory=data_dir,
        num_tokens=num_tokens,
        batch_size=micro_batch_size,
        seq_len=seq_len,
        sharding=dp_sharding,
        shuffle=True,
    )

    print(dataset[0].shape)

    if jax.process_index() == 0:
        wandb.init()

    if save_dir is not None:
        if save_interval is None:
            logging.info("Not saving checkpoints.")
        else:
            logging.warning(
                f"Saving checkpoints every {save_interval * batch_size * seq_len} tokens."
            )
            ckpt_mngr = ocp.CheckpointManager(
                save_dir,
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=1 if not save_all else None
                ),
            )

    infos = []
    for step in tqdm.trange(0, num_steps + 1):
        batch = dataset[step % len(dataset)]
        state, info = train_step(state, batch)
        infos.append(info)
        if step % grad_accum_steps == 0:
            meta_step = step // grad_accum_steps
            if meta_step % 100 == 0:
                log_all(infos, meta_step)
                infos = []
            if (
                save_dir is not None
                and save_interval is not None
                and meta_step % save_interval == 0
            ):
                print(ckpt_mngr.save(meta_step, args=flax_orbax.SaveArgs(state)))
                ckpt_mngr.check_for_errors()
                ckpt_mngr.wait_until_finished()


def log_all(infos: list[dict[str, jax.Array]], step: int):
    infos = jax.device_get(infos)
    infos = [flax.traverse_util.flatten_dict(info, sep="/") for info in infos]
    avg_info = {k: np.mean([info[k] for info in infos]) for k in infos[0].keys()}
    min_info = {
        f"min/{k}": np.min([info[k] for info in infos]) for k in infos[0].keys()
    }
    max_info = {
        f"max/{k}": np.max([info[k] for info in infos]) for k in infos[0].keys()
    }
    if jax.process_index() == 0:
        wandb.log({**avg_info, **min_info, **max_info}, step=step)


if __name__ == "__main__":
    tyro.cli(main)
