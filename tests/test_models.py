import numpy as np
import torch
import pytest
from models import (
    ResidualMLP,
    Autoencoder,
    create_residual_model,
    create_autoencoder,
    count_parameters
)


class TestModels:
    def test_residual_mlp_creation(self):
        model = ResidualMLP(input_dim=10, output_dim=4, hidden_dims=[64, 32])
        assert count_parameters(model) > 0

    def test_residual_mlp_forward(self):
        model = ResidualMLP(input_dim=10, output_dim=4, hidden_dims=[64, 32])
        x = torch.randn(32, 10)
        out = model(x)

        assert out.shape == (32, 4)
        assert torch.all(torch.isfinite(out))

    def test_residual_mlp_different_sizes(self):
        for hidden_dims in [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32]]:
            model = ResidualMLP(input_dim=10, output_dim=4, hidden_dims=hidden_dims)
            x = torch.randn(16, 10)
            out = model(x)
            assert out.shape == (16, 4)

    def test_autoencoder_creation(self):
        model = Autoencoder(input_dim=16, hidden_dims=[64, 32], latent_dim=8)
        assert count_parameters(model) > 0

    def test_autoencoder_forward(self):
        model = Autoencoder(input_dim=16, hidden_dims=[64, 32], latent_dim=8)
        x = torch.randn(32, 16)
        recon = model(x)

        assert recon.shape == (32, 16)
        assert torch.all(torch.isfinite(recon))

    def test_autoencoder_encode_decode(self):
        model = Autoencoder(input_dim=16, hidden_dims=[64, 32], latent_dim=8)
        x = torch.randn(32, 16)

        latent = model.encode(x)
        assert latent.shape == (32, 8)

        recon = model.decode(latent)
        assert recon.shape == (32, 16)

    def test_create_residual_model(self):
        for model_type in ['mlp']:
            model = create_residual_model(model_type, input_dim=10, output_dim=4)
            x = torch.randn(8, 10)
            out = model(x)
            assert out.shape == (8, 4)

    def test_create_autoencoder(self):
        for model_type in ['mlp', 'vae']:
            model = create_autoencoder(model_type, input_dim=16, latent_dim=8)
            x = torch.randn(8, 16)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            assert out.shape == (8, 16)

    def test_model_parameters_count(self):
        model = ResidualMLP(input_dim=10, output_dim=4, hidden_dims=[128, 64, 32])
        n_params = count_parameters(model)
        assert n_params > 1000
        assert n_params < 1000000

    def test_model_training_mode(self):
        model = ResidualMLP(input_dim=10, output_dim=4)
        model.train()
        assert model.training == True

        model.eval()
        assert model.training == False

    def test_gradient_flow(self):
        model = ResidualMLP(input_dim=10, output_dim=4, hidden_dims=[32, 16])
        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 4)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])