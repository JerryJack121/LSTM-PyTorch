import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNN_model, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 會用全0的 state
        out = self.out(r_out)
        return out