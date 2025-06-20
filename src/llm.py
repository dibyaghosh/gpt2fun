import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import src.transformer


class Embed(nn.Embed):
    def setup(self):
        self.embedding = self.param(
            "weight",
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype,
        )


class LLM(nn.Module):
    transformer: src.transformer.Transformer
    width: int
    vocab_size: int
    max_seq_len: int
    dtype: jnp.dtype

    def setup(self):
        self.wte = Embed(
            self.vocab_size,
            self.width,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.wpe = Embed(
            self.max_seq_len,
            self.width,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.ln_f = src.transformer.LayerNorm(self.width)

    def __call__(self, tokens: Int[Array, "batch seq_len"], train: bool):
        batch, seq_len = tokens.shape
        vocab_embeddings = self.wte(tokens)
        pos_embeddings = self.wpe(jnp.arange(seq_len))
        x = vocab_embeddings + pos_embeddings

        x = self.transformer(x, train)
        x = self.ln_f(x)
        logits = self.wte.attend(x)
        assert logits.dtype == self.dtype
        return logits
