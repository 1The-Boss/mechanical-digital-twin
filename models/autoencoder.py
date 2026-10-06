import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: List[int] = None,
        latent_dim: int = 8,
        activation: str = 'relu'
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU

        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(act_fn())
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(act_fn())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: List[int] = None,
        latent_dim: int = 8,
        activation: str = 'relu'
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(act_fn())
            prev_dim = hidden_dim

        self.encoder_base = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(act_fn())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        sequence_length: int = 50
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )

        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_lstm = nn.LSTM(
            latent_dim, hidden_dim, num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.fc_latent(h_n[-1])
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_seq = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.decoder_lstm(z_seq)
        return self.fc_out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def create_autoencoder(
    model_type: str = 'mlp',
    input_dim: int = 16,
    **kwargs
) -> nn.Module:
    if model_type == 'mlp':
        return Autoencoder(input_dim, **kwargs)
    elif model_type == 'vae':
        return VariationalAutoencoder(input_dim, **kwargs)
    elif model_type == 'lstm':
        return LSTMAutoencoder(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown autoencoder type: {model_type}")


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def compute_reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        recon = model(x)
        error = torch.mean((x - recon) ** 2, dim=1)
    return error


def compute_anomaly_scores(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str = 'cpu') -> np.ndarray:
    model.eval()
    model.to(device)
    scores = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            recon = model(x)
            error = torch.mean((x - recon) ** 2, dim=1)
            scores.append(error.cpu().numpy())

    return np.concatenate(scores) if scores else np.array([])