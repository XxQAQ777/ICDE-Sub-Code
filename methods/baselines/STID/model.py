import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + residual


class STID(nn.Module):
    """
    STID baseline adapted to TrafficFM data.

    Input:
        history_data: [B, L, N, C]
            C=3: traffic value, time_of_day, day_of_week

    Output:
        prediction: [B, H, N, 1]
            normalized traffic prediction, same convention as TrafficFM.
    """

    def __init__(
        self,
        num_nodes,
        input_len=144,
        output_len=144,
        input_dim=3,
        embed_dim=32,
        node_dim=32,
        temp_dim_tid=32,
        temp_dim_diw=32,
        time_of_day_size=288,
        day_of_week_size=7,
        num_layer=3,
        dropout=0.15,
        if_node=True,
        if_time_in_day=True,
        if_day_in_week=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.if_node = if_node
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week

        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=input_dim * input_len,
            out_channels=embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        hidden_dim = embed_dim

        if if_node:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
            nn.init.xavier_uniform_(self.node_emb)
            hidden_dim += node_dim

        if if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(time_of_day_size, temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            hidden_dim += temp_dim_tid

        if if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(day_of_week_size, temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            hidden_dim += temp_dim_diw

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(hidden_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layer)
            ]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def _time_index(self, raw_time):
        # raw_time usually lies in [0, 1). Convert to 0..287.
        idx = (raw_time * self.time_of_day_size).long()
        return torch.clamp(idx, 0, self.time_of_day_size - 1)

    def _day_index(self, raw_day):
        # Support both normalized [0, 1] and integer-like [0, 6] day encodings.
        if raw_day.max() <= 1.0:
            idx = (raw_day * self.day_of_week_size).long()
        else:
            idx = raw_day.long()
        return torch.clamp(idx, 0, self.day_of_week_size - 1)

    def forward(self, history_data):
        batch_size = history_data.shape[0]

        input_data = history_data[..., : self.input_dim]

        # [B, L, N, C] -> [B, N, L, C] -> [B, N, L*C] -> [B, L*C, N, 1]
        x = input_data.transpose(1, 2).contiguous()
        x = x.view(batch_size, self.num_nodes, self.input_len * self.input_dim)
        x = x.transpose(1, 2).unsqueeze(-1)

        time_series_emb = self.time_series_emb_layer(x)

        hidden = [time_series_emb]

        if self.if_node:
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            node_emb = node_emb.transpose(1, 2).unsqueeze(-1)
            hidden.append(node_emb)

        if self.if_time_in_day:
            raw_tid = history_data[:, -1, :, 1]
            tid_idx = self._time_index(raw_tid)
            tid_emb = self.time_in_day_emb[tid_idx]
            tid_emb = tid_emb.transpose(1, 2).unsqueeze(-1)
            hidden.append(tid_emb)

        if self.if_day_in_week:
            raw_diw = history_data[:, -1, :, 2]
            diw_idx = self._day_index(raw_diw)
            diw_emb = self.day_in_week_emb[diw_idx]
            diw_emb = diw_emb.transpose(1, 2).unsqueeze(-1)
            hidden.append(diw_emb)

        hidden = torch.cat(hidden, dim=1)
        hidden = self.encoder(hidden)

        prediction = self.regression_layer(hidden)
        return prediction
