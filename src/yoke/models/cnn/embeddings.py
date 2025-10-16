"""Various embeddings intended for use with image-based LodeRunner backbones."""

from typing import Optional

import torch
import torch.nn.functional as F


class ParallelVarEmbed(torch.nn.Module):
    """Parallel variable embedding.

    Variable to Embedding with multiple variables in a single kernel.

    Args:
        max_vars (int): Maximum number of variables
        img_size ((int, int)): Image size
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    """

    def __init__(
        self,
        max_vars: int = 8,
        img_size: tuple[int, int] = (128, 128),
        embed_dim: int = 64,
        norm_layer: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialization for parallel embedding."""
        super().__init__()

        self.max_vars = max_vars
        self.img_size = img_size
        self.embed_dim = embed_dim

        # Prepare parameters for variable embeddings.
        self._conv_template = torch.nn.Conv2d(
            in_channels=max_vars,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=None,
        )

        # Layer normalization
        self.norm = norm_layer(self.embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x: torch.Tensor, in_vars: torch.Tensor) -> torch.Tensor:
        """Forward method of the Parallel Embedding.

        NOTE: `in_vars` should be a (1, C) tensor of integers where the
        integers correspond to the variables present in the channels of
        `x`. This implies that every sample in the batch represented by `x`
        must correspond to the same variables.
        """
        proj = F.conv2d(
            x,
            weight=self._conv_template.weight[:, in_vars],
            bias=self._conv_template.bias,
            stride=self._conv_template.stride,
            padding=self._conv_template.padding,
        )

        # Normalize the layer output
        proj = self.norm(proj)

        return proj


class TimeEmbed(torch.nn.Module):
    """Temporal encoding/embedding.

    This embedding is used to help track/tag each entry of a batch by it's
    corresponding lead time. After variable aggregation and position encoding,
    temporal encoding is added to patch tokens.

    This embedding consists of a single 1D linear embedding of lead-times per
    sample in the batch to the embedding dimension.

    NOTE: Entries for image_size and patch_size should divide each other evenly.

    Args:
        embed_dim (int): Embedding dimension, must be divisible by 2.

    Forward method args:
        lead_times (torch.Tensor): lead_times.shape = (B,). Forecasting lead
                                   times of each element of the batch.

    """

    def __init__(self, embed_dim: int) -> None:
        """Initialization for temporal embedding."""
        super().__init__()

        self.lead_time_embed = torch.nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor, lead_times: torch.Tensor) -> torch.Tensor:
        """Forward method for temporal embedding."""
        # The input tensor is shape:
        #  (B, D, Y, X)=(B, embed_dim, Y, X)

        # Add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        x = x + lead_time_emb[..., None, None]  # B, D, Y, X

        return x
