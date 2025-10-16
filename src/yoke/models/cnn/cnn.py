"""CNN-based architectures for multi-channel image to multi-channel image networks."""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn.functional as F

from yoke.models.cnn.unet import UNet
from yoke.models.cnn.embeddings import ParallelVarEmbed, TimeEmbed


class ChannelDecoder(torch.nn.Module):
    """Spatial channel decoder.

    Decode channels of a spatial image from embedding space to channel space.

    Args:
        max_vars (int): Maximum number of variables
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
    """

    def __init__(
        self,
        max_vars: int = 8,
        embed_dim: int = 1,
        norm_layer: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialization for channel decoder."""
        super().__init__()

        self.max_vars = max_vars
        self.embed_dim = embed_dim

        # Prepare parameters for channel decoder.
        self._conv_template = torch.nn.Conv2d(
            in_channels=embed_dim,
            out_channels=max_vars,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=None,
        )

        # Layer normalization
        self.norm = norm_layer(self.embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x: torch.Tensor, out_vars: torch.Tensor) -> torch.Tensor:
        """Forward method of the channel decoder.

        NOTE: `out_vars` should be a (1, C) tensor of integers where the
        integers correspond to the variables present in the channels of
        `x`. This implies that every sample in the batch represented by `x`
        must correspond to the same variables.
        """
        proj = F.conv2d(
            x,
            weight=self._conv_template.weight[out_vars],
            bias=self._conv_template.bias,
            stride=self._conv_template.stride,
            padding=self._conv_template.padding,
        )

        # Normalize the layer output
        proj = self.norm(proj)

        return proj


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
        self.unet = UNet(in_channels=self.embed_dim, out_channels=self.embed_dim)

        # Define the decoder.
        self.decoder = ChannelDecoder(embed_dim=self.embed_dim, max_vars=self.max_vars)

    def forward(
        self,
        x: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for LodeRunner."""
        # Embed input to eliminate shape dependence on in_vars.
        x = self.parallel_embed(x, in_vars)

        # Encode temporal information.
        x = self.temporal_encoding(x, lead_times)

        # Pass through backbone.
        x = self.unet(x)

        # Decode only entries corresponding to out_vars for loss
        x = self.decoder(x, out_vars=out_vars)

        return x
