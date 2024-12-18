import math
from dataclasses import dataclass

import torch
from attrs import frozen

from optexp.models.model import Model
from optexp.models.initiliazation import InitializationStrategy


@frozen
class Transformer(Model):

    n_layers: int
    n_head: int
    d_model: int
    sequence_length: int
    # n_class: int
    d_mlp: int | None = None

    p_residual_dropout: float = 0.1
    p_attention_dropout: float = 0.1
    p_embedding_dropout: float = 0.1

    is_autoregressive: bool = True
    initialization: InitializationStrategy = None

    def load_model(
        self, input_shape: torch.Size, output_shape: torch.Size
    ) -> torch.nn.Module:

        model = BaseTransformer(
            self.n_layers,
            self.n_head,
            self.d_model,
            self.d_mlp,
            self.sequence_length,
            output_shape,
            self.p_residual_dropout,
            self.p_attention_dropout,
            self.p_embedding_dropout,
            self.is_autoregressive,
        )

        if self.initialization is not None:
            model = self.initialization.initialize(model)

        return model


class BaseTransformer(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_head: int,
        d_model: int,
        d_mlp: int,
        sequence_length: int,
        n_class: int,
        p_residual_dropout: float = 0.1,
        p_attention_dropout: float = 0.1,
        p_embedding_dropout: float = 0.1,
        is_autoregressive: bool = True,
        final_ln: bool = True,
    ):
        super().__init__()

        if d_mlp is None:
            d_mlp = 4 * d_model

        self.is_autoregressive = is_autoregressive

        positional_encodings = self.get_positional_embeddings(d_model, sequence_length)
        self.register_buffer("positional_encodings", positional_encodings)
        self.embeddings = self.get_embedding_layer(d_model, n_class)
        self.embedding_dropout = torch.nn.Dropout(p_embedding_dropout)
        self.prediction_layer = self.get_prediction_layer(d_model, n_class)

        encoder_layer = self.get_encoder_layer(
            d_model, n_head, d_mlp, p_residual_dropout, p_attention_dropout
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=final_ln
        )

    def get_encoder_layer(
        self,
        d_model: int,
        n_head: int,
        d_mlp: int,
        p_residual_dropout: float = 0.1,
        p_attention_dropout: float = 0.1,
    ):
        norm1, norm2 = self.get_normalize_blocks(d_model)
        attn = self.get_attention_block(d_model, n_head, p_attention_dropout)
        mlp = self.get_fully_connected_block(d_model, d_mlp, p_residual_dropout)

        class EncoderLayer(torch.nn.Module):
            def forward(self, x):
                x = x + attn(norm1(x))
                x = x + mlp(norm2(x))
                return x

        return EncoderLayer()

    def get_normalize_blocks(self, d_model: int):
        return torch.nn.LayerNorm(d_model), torch.nn.LayerNorm(d_model)

    def get_attention_block(self, d_model: int, n_head: int, p_drop: float):
        return torch.nn.MultiheadAttention(d_model, n_head, p_drop)

    def get_fully_connected_block(
        self, d_model: int, d_mlp: int, p_drop: float
    ) -> torch.nn.Module:
        lin1 = torch.nn.Linear(d_model, d_mlp)
        activation = torch.nn.GELU(approximate="tanh")
        lin2 = torch.nn.Linear(d_mlp, d_model)
        dropout = torch.nn.Dropout(p_drop)
        return torch.nn.Sequential(lin1, activation, lin2, dropout)

    def get_positional_embeddings(self, d_model: int, sequence_length: int):
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(sequence_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def get_prediction_layer(self, d_model: int, n_class: int):
        return torch.nn.Linear(d_model, n_class)

    def get_embedding_layer(self, d_model: int, n_class: int):
        return torch.nn.Embedding(n_class, d_model)

    def get_attention_mask(self, sequence_length):
        mask = (
            torch.triu(torch.ones(sequence_length, sequence_length)) == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    # TODO valudate which index goes where
    def forward(self, x):
        x = self.embedding_dropout(
            self.embeddings(x) + self.positional_embeddings[: x.shape[0]]
        )
        mask = self.get_attention_mask(x.shape[0])
        x = self.encoder(x, mask=mask, is_causal=self.is_autoregressive)
        x = self.prediction_layer(x)
        return x
