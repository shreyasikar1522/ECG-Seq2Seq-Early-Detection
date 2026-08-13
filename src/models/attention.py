import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(
        self,
        encoder_hidden_size,
        decoder_hidden_size
    ):

        super().__init__()

        self.attn = nn.Linear(
            encoder_hidden_size + decoder_hidden_size,
            decoder_hidden_size
        )

        self.v = nn.Linear(
            decoder_hidden_size,
            1,
            bias=False
        )

    def forward(
        self,
        decoder_hidden,
        encoder_outputs
    ):

        # encoder_outputs:
        # (batch, seq_len, encoder_hidden)

        batch_size = encoder_outputs.size(0)

        seq_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(1)

        decoder_hidden = decoder_hidden.repeat(
            1,
            seq_len,
            1
        )

        energy = torch.tanh(

            self.attn(

                torch.cat(

                    (
                        decoder_hidden,
                        encoder_outputs
                    ),

                    dim=2

                )

            )

        )

        attention = self.v(
            energy
        ).squeeze(2)

        attention_weights = torch.softmax(
            attention,
            dim=1
        )

        context = torch.bmm(

            attention_weights.unsqueeze(1),

            encoder_outputs

        )

        context = context.squeeze(1)

        return context, attention_weights