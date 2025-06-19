import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import src.transformer


class LLM(nn.Module):
    transformer: src.transformer.Transformer
    width: int
    vocab_size: int
    max_seq_len: int
    dtype: jnp.dtype

    def setup(self):
        self.embedding = nn.Embed(
            self.vocab_size,
            self.width,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.ln_final = src.transformer.LayerNorm(self.width)
        self.pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_seq_len, self.width),
        )

    def __call__(self, tokens: Int[Array, "batch seq_len"], train: bool):
        batch, seq_len = tokens.shape
        vocab_embeddings = self.embedding(tokens)
        x = vocab_embeddings + self.pos_emb[:, :seq_len, :].astype(self.dtype)
        assert x.dtype == self.dtype
        x = self.transformer(x, train)
        assert x.dtype == self.dtype
        x = self.ln_final(x)
        logits = self.embedding.attend(x)
        assert logits.dtype == self.dtype
        return logits
