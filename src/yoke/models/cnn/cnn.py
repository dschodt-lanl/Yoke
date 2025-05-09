"""CNN-based architectures for multi-channel image to multi-channel image networks."""

from collections.abc import Iterable
from typing import Any

import torch

from yoke.models.cnn.unet import UNet
from yoke.models.cnn.embeddings import ParallelVarEmbed, TimeEmbed


class LodeRunnerCNN(torch.nn.Module):
    """LodeRunner neural network.

    CNN-based LodeRunner with variable embedding.

    Args:
        default_vars (list[str]): List of default variables to be used for training
        image_size (tuple[int, int]): Height and width, in pixels, of input image.
        embed_dim (int): Initial embedding dimension.
    """

    def __init__(
        self,
        default_vars: list[str],
        image_size: Iterable[int, int] = (1120, 800),
        embed_dim: int = 128,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialization for class."""
        super().__init__()

        self.default_vars = default_vars
        self.max_vars = len(self.default_vars)
        self.image_size = image_size
        self.embed_dim = embed_dim

        # Define learnable variable channel embedding.
        self.parallel_embed = ParallelVarEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Define learnable time encoding.
        self.temporal_encoding = TimeEmbed(self.embed_dim)

        # Define network backbone.
        self.backbone = UNet(in_channels=self.embed_dim, out_channels=self.max_vars)

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for LodeRunner."""
        # WARNING!: Most likely the `in_vars` and `out_vars` need to be tensors
        # of integers corresponding to variables in the `default_vars` list.

        # Embed input to eliminate shape dependence on in_vars.
        x = self.parallel_embed(x, in_vars)

        # Encode temporal information.
        x = self.temporal_encoding(x, lead_times)

        # Pass through backbone.
        x = self.backbone(x)

        # Select only entries corresponding to out_vars for loss
        preds = x[:, out_vars]

        return preds
