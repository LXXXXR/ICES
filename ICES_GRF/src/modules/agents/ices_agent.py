import torch
import torch.nn as nn
import torch.nn.functional as F


class ICESAgent(nn.Module):
    def __init__(self, input_shape, args, nonlinear=False):
        super(ICESAgent, self).__init__()
        self.args = args
        self.nonlinear = nonlinear

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True,
        )
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        if nonlinear:
            self.tanh = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0)

        hidden_state = hidden_state.contiguous()
        input_shape = inputs.shape

        if len(input_shape) == 2:
            x = F.relu(self.fc1(inputs))
            x = x.unsqueeze(1)
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim).unsqueeze(0)
            gru_out, _ = self.rnn(x, h_in)
            gru_out = gru_out.reshape(-1, self.args.rnn_hidden_dim)
            q = self.fc2(gru_out)

        elif len(input_shape) == 4:
            inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
            inputs = inputs.reshape(-1, inputs.shape[-1])

            x = F.relu(self.fc1(inputs))
            x = x.reshape(-1, input_shape[2], x.shape[-1])

            gru_out, _ = self.rnn(x, hidden_state.to(x.device))
            gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
            q = self.fc2(gru_out_c)

            q = q.reshape(-1, gru_out.shape[1], q.shape[-1])
            q = q.reshape(-1, input_shape[1], q.shape[-2], q.shape[-1]).permute(
                0, 2, 1, 3
            )

        if self.nonlinear:
            q = self.tanh(q)

        return q, gru_out, None


class ICESCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(ICESCritic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True,
        )
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0)

        hidden_state = hidden_state.contiguous()
        input_shape = inputs.shape

        if len(input_shape) == 2:
            x = F.relu(self.fc1(inputs))
            x = x.unsqueeze(1)
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim).unsqueeze(0)
            gru_out, _ = self.rnn(x, h_in)
            gru_out = gru_out.reshape(-1, self.args.rnn_hidden_dim)
            q = self.fc2(gru_out)

        elif len(input_shape) == 4:
            inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
            inputs = inputs.reshape(-1, inputs.shape[-1])

            x = F.relu(self.fc1(inputs))
            x = x.reshape(-1, input_shape[2], x.shape[-1])

            gru_out, _ = self.rnn(x, hidden_state.to(x.device))
            gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
            q = self.fc2(gru_out_c)

            q = q.reshape(-1, gru_out.shape[1], q.shape[-1])
            q = q.reshape(-1, input_shape[1], q.shape[-2], q.shape[-1]).permute(
                0, 2, 1, 3
            )

        return q, gru_out, None
