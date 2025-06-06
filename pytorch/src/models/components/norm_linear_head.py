# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinearHead(nn.Module):
    """A classification head with layer normalization followed by linear projection.

    This head applies layer normalization to the input features before projecting them to the
    output dimension. Optionally normalizes the output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        normalize_output: bool = False,
    ) -> None:
        """Initialize the NormLinearHead.

        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
            normalize_output: Whether to normalize the output
        """
        super().__init__()
        self.normalize_output = normalize_output
        self.layers = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head.

        Args:
            x: Input tensor of shape (batch_size, embed_dim)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.layers(x)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        return x
