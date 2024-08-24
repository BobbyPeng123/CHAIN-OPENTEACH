import torch.nn as nn
import utils

from agent.networks.s4 import S4Block as S4


class S4Net(nn.Module):
    def __init__(
        self,
        model_dim,
        nlayers,
        prenorm=True,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(S4(model_dim, dropout=0.0, transposed=True))
            self.norms.append(nn.LayerNorm(model_dim))
            self.dropouts.append(nn.Dropout(0.0))

    def forward(self, x, lengths=None):
        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                # z = norm(z)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                # x = norm(x)
        x = x.transpose(-1, -2)
        return x

    def default_state(self, *batch_shape, device=None):
        return {
            i: layer.default_state(*batch_shape, device=device)
            for i, layer in enumerate(self.layers)
        }

    def step(self, x, state_dict):
        # x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for i, (layer, norm, dropout) in enumerate(
            zip(self.layers, self.norms, self.dropouts)
        ):
            z = x
            if self.prenorm:
                z = norm(z)
            z, state_dict[i] = layer.step(z, state_dict[i])
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x)
        return x, state_dict

    def reset(self):
        for i in range(len(self.layers)):
            self.layers[i].setup_step()


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1) -> None:
        super().__init__()
        self.model = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        x, _ = self.model(x)
        return x


class RNNEncoder(nn.Module):
    def __init__(self, obs_shape, rnn_type="lstm") -> None:
        super().__init__()

        # assert len(obs_shape) == 2
        self.rnn_dim = 128
        self.repr_dim = 512
        self.nlayers = 4
        self.embed = nn.Linear(obs_shape[0], self.rnn_dim)
        self.out_layer = nn.Linear(self.rnn_dim, self.repr_dim)
        if rnn_type == "lstm":
            self.rnn = LSTMNet(self.rnn_dim, self.rnn_dim, 1)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(obs_shape[1], self.rnn_dim, 1)
        elif rnn_type == "s4":
            self.rnn = S4Net(self.rnn_dim, self.nlayers)
        elif rnn_type == "mamba":
            self.rnn = Mamba(512, 512)

    def forward(self, x):
        x = self.embed(x)
        x = self.rnn(x)
        x = self.out_layer(x)
        return x[:, -1:]
