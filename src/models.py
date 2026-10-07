# Three networks from Section 4: LSTM-128, Fusion (parallel), Cascade (sequential).

from dataclasses import dataclass

import torch
import torch.nn as nn

import config


class PureLSTM(nn.Module):
    # LSTM-128: 128 hidden units, 1 layer, dropout 0.2.
    def __init__(self, n_features, hidden_size=128, output_size=5,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, 60 days, 5 features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # keep only the last day
        return self.fc(out)                 # 5 predicted log-returns


class FusionCNNLSTM(nn.Module):
    # Parallel: CNN and LSTM both see the raw 60-day window, then we concatenate.
    def __init__(self, n_features, hidden_size=64, output_size=5,
                 cnn_filters=32, num_layers=1, dropout=0.3):
        super().__init__()
        # Conv1d wants (batch, channels, time). padding=1 keeps length 60 (Section 4.3).
        # Then we pool and average over time so the CNN becomes one vector.
        self.conv1 = nn.Conv1d(n_features, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)
        self.lstm = nn.LSTM(n_features, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + cnn_filters, output_size)

    def forward(self, x):
        cnn_y = x.permute(0, 2, 1)          # (batch, 5, 60)
        cnn_y = self.dropout_cnn(self.relu(self.pool(self.conv1(cnn_y))))
        cnn_y = torch.mean(cnn_y, dim=2)    # average over time
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out[:, -1, :])
        return self.fc(torch.cat((lstm_out, cnn_y), dim=1))


class CascadeCNNLSTM(nn.Module):
    # Sequential: CNN first, then LSTM reads the CNN maps (not raw prices).
    def __init__(self, n_features, hidden_size=64, output_size=5,
                 cnn_filters=32, num_layers=1, dropout=0.3):
        super().__init__()
        # No padding: after kernel 3 and pool 2, length becomes 29.
        self.conv1 = nn.Conv1d(n_features, cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.pool(self.conv1(x)))
        x = x.permute(0, 2, 1)              # (batch, time, filters)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


@dataclass
class ModelSpec:
    factory: type
    kwargs: dict
    learning_rate: float
    weight_decay: float = 0.0


# Names used in train.py: lstm128, fusion, cascade.
MODEL_REGISTRY = {
    "lstm128": ModelSpec(
        factory=PureLSTM,
        kwargs=dict(hidden_size=128, num_layers=1, dropout=0.2),
        learning_rate=config.LEARNING_RATE_LSTM,
        weight_decay=config.WEIGHT_DECAY_LSTM,
    ),
    "fusion": ModelSpec(
        factory=FusionCNNLSTM,
        kwargs=dict(hidden_size=64, cnn_filters=32, num_layers=1, dropout=0.3),
        learning_rate=config.LEARNING_RATE_HYBRID,
        weight_decay=config.WEIGHT_DECAY_HYBRID,
    ),
    "cascade": ModelSpec(
        factory=CascadeCNNLSTM,
        kwargs=dict(hidden_size=64, cnn_filters=32, num_layers=1, dropout=0.3),
        learning_rate=config.LEARNING_RATE_HYBRID,
        weight_decay=config.WEIGHT_DECAY_HYBRID,
    ),
}

MODEL_NAMES = list(MODEL_REGISTRY)


def build_model(name, n_features=len(config.FEATURES),
                output_size=len(config.FEATURES)):
    # New untrained net. Never loads old weights unless you do it yourself.
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
    spec = MODEL_REGISTRY[name]
    model = spec.factory(n_features=n_features, output_size=output_size,
                         **spec.kwargs)
    return model, spec


def build_optimizer(model, spec):
    return torch.optim.Adam(model.parameters(), lr=spec.learning_rate,
                            weight_decay=spec.weight_decay)
