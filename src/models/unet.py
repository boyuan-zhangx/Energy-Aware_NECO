"""U-Net architecture used in the MUAD notebook."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TVF


class DoubleConv(nn.Module):
    """Two convolution, batch-normalization, and ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InConv(nn.Module):
    """Initial notebook U-Net block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Down-sampling block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """Up-sampling block with skip concatenation."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.bilinear = bilinear
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x1 = TVF.resize(
                x1,
                size=[2 * x1.size()[2], 2 * x1.size()[3]],
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        else:
            x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = TVF.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """Final one-by-one classifier."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Compact U-Net with decoder dropout for uncertainty methods."""

    def __init__(
        self,
        classes: int,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.inc = InConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)
        self.up1 = Up(base_channels * 16, base_channels * 4, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 2, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.outc = OutConv(base_channels, classes)

    def decoder_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.dropout(self.up1(x5, x4))
        x = self.dropout(self.up2(x, x3))
        x = self.dropout(self.up3(x, x2))
        x = self.dropout(self.up4(x, x1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outc(self.decoder_features(x))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load notebook checkpoints and the earlier translated-project checkpoints.

        The notebook architecture keeps an unused ConvTranspose2d inside every
        bilinear up block, so those weights are present in notebook checkpoints.
        Earlier project checkpoints omitted them and used shorter module names;
        those keys are mapped here to keep old local artifacts loadable.
        """

        own_state = super().state_dict()
        converted = {}
        for key, value in state_dict.items():
            converted[_project_key_to_notebook_key(key)] = value

        if strict and self._uses_notebook_bilinear_upsampling():
            for key, value in own_state.items():
                if _is_unused_upconv_key(key) and key not in converted:
                    converted[key] = value

        return super().load_state_dict(converted, strict=strict, assign=assign)

    def _uses_notebook_bilinear_upsampling(self) -> bool:
        return all(getattr(getattr(self, f"up{idx}"), "bilinear", False) for idx in range(1, 5))


def _project_key_to_notebook_key(key: str) -> str:
    if key.startswith("inc.net."):
        return key.replace("inc.net.", "inc.conv.conv.", 1)

    for idx in range(1, 5):
        down_prefix = f"down{idx}.net.1.net."
        if key.startswith(down_prefix):
            return key.replace(down_prefix, f"down{idx}.mpconv.1.conv.", 1)

        up_prefix = f"up{idx}.conv.net."
        if key.startswith(up_prefix):
            return key.replace(up_prefix, f"up{idx}.conv.conv.", 1)

    return key


def _is_unused_upconv_key(key: str) -> bool:
    return any(key in {f"up{idx}.up.weight", f"up{idx}.up.bias"} for idx in range(1, 5))
