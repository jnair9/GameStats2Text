import torch
import torch.nn as nn


class StatsEncoder(nn.Module):
    """
    A simple feed-forward encoder that transforms raw game statistics
    into a dense embedding vector. Uses fully connected layers with
    ReLU activations and dropout for regularization.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim (int): Number of input features (statistical metrics).
            hidden_dims (list[int]): List of hidden layer sizes.
            output_dim (int): Dimension of the output embedding.
            dropout (float): Dropout probability between layers.
        """
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: takes a batch of raw stats and returns embeddings.

        Args:
            x (torch.Tensor): Batch of input features of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, output_dim).
        """
        return self.encoder(x)
