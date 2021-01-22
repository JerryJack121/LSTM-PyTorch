import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNN_model, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.drop = nn.Dropout(0.5)
        self.out = nn.Sequential(
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn1(x, None)  # None 表示 hidden state 會用全0的 state
        d_out = self.drop(r_out)
        r_out, (h_n, h_c) = self.rnn2(d_out, None)
        d_out = self.drop(r_out)
        out = self.out(d_out)
        return out