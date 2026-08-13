import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Generic Encoder supporting:
        - RNN
        - GRU
        - LSTM
        - BiLSTM
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        encoder_type="bilstm",
        dropout=0.2
    ):

        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bidirectional = (
            self.encoder_type == "bilstm"
        )

        num_directions = 2 if self.bidirectional else 1

        if self.encoder_type == "rnn":

            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )

        elif self.encoder_type == "gru":

            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )

        elif self.encoder_type == "lstm":

            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )

        elif self.encoder_type == "bilstm":

            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )

        else:

            raise ValueError(
                f"Unknown encoder type: {encoder_type}"
            )

        self.num_directions = num_directions

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        encoder_outputs :
            (batch, seq_len, hidden)
            or
            (batch, seq_len, hidden*2) for BiLSTM

        hidden :
            Final hidden state

        cell :
            Final cell state (None for RNN/GRU)
        """

        if self.encoder_type in ["lstm", "bilstm"]:

            encoder_outputs, (hidden, cell) = self.rnn(x)

        else:

            encoder_outputs, hidden = self.rnn(x)

            cell = None

        return encoder_outputs, hidden, cell
    def output_size(self):
        """
        Returns the dimension of each encoder output.
        """

        if self.bidirectional:
            return self.hidden_size * 2

        return self.hidden_size
if __name__ == "__main__":

    x = torch.randn(
        64,
        98,
        1
    )

    encoder = Encoder(
        encoder_type="bilstm"
    )

    outputs, hidden, cell = encoder(x)

    print("Outputs :", outputs.shape)
    print("Hidden  :", hidden.shape)

    if cell is not None:
        print("Cell    :", cell.shape)