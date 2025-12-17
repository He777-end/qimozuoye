# src/models.py

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    用于基线模型的简单 MLP：
    输入：展平后的特征向量
    输出：该分 player1 赢的 logit（之后经 sigmoid 得概率）
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)  # (batch, 1)
        return logits.squeeze(-1)


class LSTMPointPredictor(nn.Module):
    """
    LSTM 序列模型：
    输入： (batch, seq_len, input_dim)
    输出： logits (batch,)
    结构：LSTM -> Dropout -> Dense
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,   # 明确的 dropout 超参
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # 可以用最后一层最后一个时间步 hidden，也可以用 out[:, -1, :]
        last_hidden = h_n[-1]               # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)       # (batch, 1)
        return logits.squeeze(-1)


class LSTMSeqPredictor(nn.Module):
    """
    多步预测版本：LSTM + Dropout + Dense(horizon)

    输入:  (batch, seq_len, input_dim)
    输出:  logits (batch, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]           # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)   # (batch, horizon)
        return logits  # 不 squeeze，保持 (batch, horizon)


class TransformerPointPredictor(nn.Module):
    """
    基于 TransformerEncoder 的单步预测模型：
    Linear(input_dim -> d_model) + TransformerEncoder + Dropout + Dense
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 直接用 (batch, seq_len, dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        out = self.encoder(x)           # (batch, seq_len, d_model)
        last_hidden = out[:, -1, :]     # 取最后一个时间步
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)   # (batch, 1)
        return logits.squeeze(-1)


class TransformerSeqPredictor(nn.Module):
    """
    基于 TransformerEncoder 的多步预测模型：
    Linear -> TransformerEncoder -> Dropout -> Dense(horizon)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        out = self.encoder(x)           # (batch, seq_len, d_model)
        last_hidden = out[:, -1, :]     # (batch, d_model)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)   # (batch, horizon)
        return logits
