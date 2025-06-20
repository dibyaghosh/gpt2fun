import dataclasses
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclasses.dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    activation: Any = partial(nn.gelu, approximate=True)

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = LayerNorm(self.config.n_embd)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = LayerNorm(self.config.n_embd)
        self.mlp = MLP(self.config)

    def __call__(self, x, train):
        x = self.perturb("input", x)
        x = x + self.attn(self.ln_1(x), train=train)
        x = self.perturb("post_attn", x)
        x = x + self.mlp(self.ln_2(x), train=train)
        x = self.perturb("post_mlp", x)
        return x


class Transformer(nn.Module):
    config: GPTConfig

    def setup(self):
        self.blocks = [Block(self.config) for _ in range(self.config.n_layer)]

    def __call__(self, x, train):
        for block in self.blocks:
            x = block(x, train)
        return x


class Sequential(nn.Module):
    layers: dict[str, nn.Module]

    def __call__(self, x, **kwargs):
        for layer in self.layers.values():
            x = self.bind(layer, kwargs)(x)
        return x

    @staticmethod
    def bind(fn, kwargs):
        import functools
        import inspect

        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return functools.partial(fn, **kwargs)


class LayerNorm(nn.Module):
    width: int
    eps: float = 1e-5

    def setup(self):
        self.weight = self.param("weight", nn.initializers.ones, (self.width,))
        self.bias = self.param("bias", nn.initializers.zeros, (self.width,))

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        var = jnp.maximum(mean2 - jnp.square(mean), 0)
        mul = jax.lax.rsqrt(var + self.eps) * self.weight
        y = (x - mean) * mul + self.bias
        return y.astype(dtype)


class Dense(nn.Module):
    input_dim: int
    output_dim: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.normal(stddev=0.02)
    bias_init: Any = nn.initializers.zeros

    def setup(self):
        self.kernel = self.param(
            "weight", self.kernel_init, (self.input_dim, self.output_dim)
        )
        if self.use_bias:
            self.bias = self.param("bias", self.bias_init, (self.output_dim,))

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        dtype = x.dtype
        y = x @ self.kernel.astype(dtype)
        if self.use_bias:
            y = y + self.bias.astype(dtype)
        return y


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        self.c_attn = Dense(
            self.config.n_embd,
            3 * self.config.n_embd,
            use_bias=self.config.bias,
        )
        self.c_proj = Dense(
            self.config.n_embd,
            self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.normal(
                stddev=0.02 / jnp.sqrt(2 * self.config.n_layer)
            ),
        )
        self.resid_dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x, train):
        B, T, C = x.shape
        x = self.perturb("input", x)
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q, k, v = self.perturb("q", q), self.perturb("k", k), self.perturb("v", v)

        q, k, v = jax.tree.map(
            lambda x: x.reshape(B, T, self.config.n_head, self.config.head_dim),
            (q, k, v),
        )
        x = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        x = x.reshape(B, T, self.config.n_embd)
        x = self.perturb("pre_proj", x)
        x = self.c_proj(x)
        return self.resid_dropout(x, deterministic=not train)


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        self.c_fc = Dense(
            self.config.n_embd, 4 * self.config.n_embd, use_bias=self.config.bias
        )
        self.c_proj = Dense(
            4 * self.config.n_embd, self.config.n_embd, use_bias=self.config.bias
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x, train):
        x = self.perturb("input", x)
        x = self.c_fc(x)
        x = self.perturb("pre_proj", x)
        x = self.config.activation(x)
        x = self.c_proj(x)
        return self.dropout(x, deterministic=not train)


if __name__ == "__main__":
    config = GPTConfig()
    model = Transformer(config)
    jax.eval_shape(
        partial(model.init, train=True), jax.random.PRNGKey(0), jnp.ones((1, 1024, 768))
    )
