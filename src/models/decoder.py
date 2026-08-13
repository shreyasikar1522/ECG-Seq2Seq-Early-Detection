import random

import torch
import torch.nn as nn

from .attention import BahdanauAttention


class Decoder(nn.Module):

    def __init__(
        self,
        encoder_hidden_size=128,
        decoder_hidden_size=64,
        output_size=1,
        num_layers=2,
        target_len=42,
        dropout=0.2
    ):

        super().__init__()

        self.target_len = target_len

        self.attention = BahdanauAttention(
            encoder_hidden_size,
            decoder_hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=encoder_hidden_size + output_size,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(
            decoder_hidden_size,
            output_size
        )

    def forward(
        self,
        hidden,
        cell,
        encoder_outputs,
        encoder_input,
        target=None,
        teacher_forcing_ratio=0.5
    ):

        batch_size = encoder_input.size(0)

        decoder_input = encoder_input[:, -1:, :]

        outputs = []

        attention_maps=[]

        for t in range(self.target_len):

            decoder_hidden = hidden[-1]

            context, attention = self.attention(

                decoder_hidden,

                encoder_outputs

            )

            attention_maps.append(attention)

            lstm_input = torch.cat(

                (
                    decoder_input,
                    context.unsqueeze(1)
                ),

                dim=2

            )

            output, (hidden, cell) = self.lstm(

                lstm_input,

                (
                    hidden,
                    cell
                )

            )

            prediction = self.fc(output)

            outputs.append(prediction)

            if (

                target is not None

                and random.random() < teacher_forcing_ratio

            ):

                decoder_input = target[:, t:t+1, :]

            else:

                decoder_input = prediction

        outputs = torch.cat(

            outputs,

            dim=1

        )

        attention_maps = torch.stack(

            attention_maps,

            dim=1

        )

        return outputs, attention_maps