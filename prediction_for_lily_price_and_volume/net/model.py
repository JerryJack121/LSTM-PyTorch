import torch.nn as nn

class RNN_modelv1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNN_modelv1, self).__init__()

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Sequential(
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, (h_n, h_c) = self.rnn1(x, None)  # None 表示 hidden state 會用全0的 state
        out = self.out(out)
        return out

class RNN_modelv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNN_modelv2, self).__init__()
        # self.dnn1 = nn.Linear(input_dim, 1024)
        # self.dnn2 = nn.Linear(1024, 512)

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=2048,
            num_layers=3,
            batch_first=True
        )
        # self.rnn2 = nn.LSTM(
        #     input_size=128,
        #     hidden_size=64,
        #     num_layers=3,
        #     batch_first=True
        # )
        
        # self.drop = nn.Dropout(0.5)
        self.out = nn.Sequential(
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        # out = self.dnn1(x)
        # out = self.dnn2(out)
        # out = self.drop(out)
        out, (h_n, h_c) = self.rnn1(x, None)  # None 表示 hidden state 會用全0的 state
        # out = self.drop(out)
        # out, (h_n, h_c) = self.rnn2(out, None)        
        out = self.out(out)
        return out