import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 4,
        hidden_dims: List[int] = None,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class ResidualTCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 4,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation

            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                    padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out


def create_residual_model(
    model_type: str = 'mlp',
    input_dim: int = 10,
    output_dim: int = 4,
    **kwargs
) -> nn.Module:
    if model_type == 'mlp':
        return ResidualMLP(input_dim, output_dim, **kwargs)
    elif model_type == 'lstm':
        return ResidualLSTM(input_dim, output_dim, **kwargs)
    elif model_type == 'tcn':
        return ResidualTCN(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        ""
    ]

    for name, param in model.named_parameters():
        lines.append(f"  {name}: {param.shape} ({param.numel():,} params)")

    return "\n".join(lines)