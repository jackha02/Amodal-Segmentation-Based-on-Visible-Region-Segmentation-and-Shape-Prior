# Lightweight U-Net reconstruction network for amodal mask refinement.
#
# Motivation
# ----------
# Xiao et al. (2021) use a pre-trained auto-encoder with a K-Means codebook
# as the shape prior for amodal mask refinement.  Kim et al. (2023) showed
# that for single-class objects with high intraclass shape variance — such as
# cucumbers of varying scale and viewpoint — this auto-encoder approach
# underperforms relative to U-Net.  The authors attribute this to two
# properties of U-Net:
#
#   1. Scale invariance: skip connections between the contraction and
#      expansion paths preserve spatial detail at multiple resolutions,
#      giving the network the ability to reconstruct instances that appear
#      at very different sizes in the image (e.g. due to different camera
#      distances or viewpoints)
#
#   2. Small-dataset efficiency: U-Net was designed for biomedical
#      segmentation where labelled data is scarce.  Its fully-convolutional
#      structure with no dense layers generalises well from small datasets
#
#   3. Speed: the end-to-end convolutional structure is faster than the
#      auto-encoder + codebook retrieval pipeline at inference time
#
# Storm drain inlets share the same physical geometry (rectangular grate)
# but can appear very different depending on camera viewpoint (oblique vs
# overhead), distance, and partial clogging geometry. The same reasoning
# that led Kim et al. to prefer U-Net over the auto-encoder therefore
# applies directly to this task
#
# Architecture
# ------------
# The input to U-Net is the coarse amodal mask M^c_a (shape: N×1×H×W).
# The output is a same-sized refined amodal mask logit (N×1×H×W).
# This replaces the auto-encoder + codebook shape prior in Module 3 of
# the original VRSP-Net pipeline (Xiao et al. 2021, Fig. 2).
#
# The network follows Ronneberger et al. (2015) exactly:
#   Encoder: 3×3 Conv → BN → ReLU (×2), then 2×2 MaxPool
#   Decoder: 2×2 bilinear upsample → concat skip → 3×3 Conv → BN → ReLU (×2)
#   Output:  1×1 Conv to single-channel logit
#
# We use a reduced channel width (base_channels=16) appropriate for the 
# small 28×28 ROI feature maps used in Detectron2's mask head
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class _DoubleConv(nn.Sequential):
    """Two consecutive 3×3 Conv → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class _Down(nn.Module):
    """2×2 MaxPool then DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)

class _Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv (decoder step)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch is the sum of upsampled channels + skip channels
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetReconstruction(nn.Module):
    """
    Lightweight U-Net for amodal mask reconstruction.

    Input  : coarse amodal mask logits  (N, 1, H, W)
    Output : refined amodal mask logits (N, 1, H, W)

    Parameters
    ----------
    base_channels : int
        Number of feature channels in the first encoder block.
        Doubles at each depth level.  Default 16 is appropriate for the
        small (28×28) ROI feature maps in Detectron2.
    depth : int
        Number of encoder/decoder stages.  Default 3 gives a bottleneck
        at H/8 × W/8, which is 3×3 for 28×28 input — a good trade-off
        between receptive field and information loss.
    """

    def __init__(self, base_channels: int = 16, depth: int = 3):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc   = _DoubleConv(1, c)           # 1 → c
        self.downs = nn.ModuleList(
            [_Down(c * 2**i, c * 2**(i + 1)) for i in range(depth)]
        )                                          # c→2c, 2c→4c, 4c→8c

        # Decoder (channels: upsampled_from_below + encoder_skip → out)
        # At decoder level i the upsampled tensor has c*2^(i+1) channels
        # (coming from the deeper level) and the encoder skip has c*2^i channels.
        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch  = c * 2**(i + 1) + c * 2**i  # upsampled + skip (different widths)
            out_ch = c * 2**i
            self.ups.append(_Up(in_ch, out_ch))

        # Final 1×1 conv to logit
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1, H, W)  coarse amodal mask (raw logits or sigmoid output)

        Returns
        -------
        (N, 1, H, W)  refined amodal mask logits
        """
        # Encoder — store skip connections
        skips = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)

        # The deepest feature map is the bottleneck; pop it off
        h = skips.pop()

        # Decoder
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)

        return self.out_conv(h)   # (N, 1, H, W)  raw logits