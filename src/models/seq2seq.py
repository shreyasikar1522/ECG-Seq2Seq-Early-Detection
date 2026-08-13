import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder_type="bilstm",
        input_size=1,
        hidden_size=64,
        output_size=1,
        num_layers=2,
        target_len=42,
        dropout=0.2
    ):

        super().__init__()

        self.encoder_type = encoder_type.lower()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.bidirectional = (
            self.encoder_type == "bilstm"
        )

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            encoder_type=encoder_type,
            dropout=dropout
        )

        encoder_output_size = self.encoder.output_size()

        self.decoder = Decoder(
            encoder_hidden_size=encoder_output_size,
            decoder_hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            target_len=target_len,
            dropout=dropout
        )

        # Bridge for BiLSTM
        if self.bidirectional:

            self.hidden_bridge = nn.Linear(
                hidden_size * 2,
                hidden_size
            )

            self.cell_bridge = nn.Linear(
                hidden_size * 2,
                hidden_size
            )

    def _bridge_bidirectional(self, state):

        """
        Convert

        (layers*2, batch, hidden)

        →

        (layers, batch, hidden)
        """

        bridged = []

        for layer in range(self.num_layers):

            forward_state = state[2 * layer]

            backward_state = state[2 * layer + 1]

            merged = torch.cat(

                (
                    forward_state,
                    backward_state
                ),

                dim=1

            )

            bridged.append(merged)

        bridged = torch.stack(
            bridged,
            dim=0
        )

        return bridged

    def forward(
        self,
        x,
        target=None,
        teacher_forcing_ratio=0.5
    ):

        encoder_outputs, hidden, cell = self.encoder(x)

        # -----------------------------
        # Bridge BiLSTM hidden states
        # -----------------------------

        if self.bidirectional:

            hidden = self._bridge_bidirectional(hidden)

            cell = self._bridge_bidirectional(cell)

            hidden = self.hidden_bridge(hidden)

            cell = self.cell_bridge(cell)

        elif cell is None:

            cell = torch.zeros_like(hidden)

        predictions, attention_maps = self.decoder(

            hidden=hidden,

            cell=cell,

            encoder_outputs=encoder_outputs,

            encoder_input=x,

            target=target,

            teacher_forcing_ratio=teacher_forcing_ratio

        )

        return predictions, attention_maps
if __name__ == "__main__":

    x = torch.randn(
        8,
        98,
        1
    )

    y = torch.randn(
        8,
        42,
        1
    )

    model = Seq2Seq(
        encoder_type="bilstm"
    )

    output = model(
        x,
        target=y
    )

    print(output.shape)