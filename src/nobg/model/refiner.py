import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    """Convolution followed by GroupNorm and SiLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )


def _resize(tensor, size):
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class SegFormerRefiner(nn.Module):
    """Improves SegFormer predictions by adding a small convolutional network on top of the features and logits."""

    def __init__(self, f1_channels):
        super().__init__()
        # H/4
        self.conv1 = ConvBlock(f1_channels, 32, kernel_size=1)
        self.conv2 = ConvBlock(32 + 1 + 3, 32)
        self.conv3 = ConvBlock(32, 32)

        # H/2
        self.conv4 = ConvBlock(3, 16, stride=2)
        self.conv5 = ConvBlock(32 + 16 + 1, 32)
        self.conv6 = ConvBlock(32, 16)

        # H x W
        self.conv7 = ConvBlock(3, 8)
        self.conv8 = ConvBlock(16 + 8 + 1, 16)
        self.conv9 = ConvBlock(16, 16)
        self.conv10 = nn.Conv2d(16, 1, kernel_size=1)
        
        nn.init.zeros_(self.conv10.weight)
        nn.init.zeros_(self.conv10.bias)

    def forward(self, f1, logits, image):
        
        quarter_size = f1.shape[-2:]
        
        x = self.conv2(torch.cat((
            self.conv1(f1), _resize(logits, quarter_size),
            _resize(image, quarter_size),
        ), dim=1))
        x = self.conv3(x)

        rgb_half = self.conv4(image)
        half_size = rgb_half.shape[-2:]  # Also handles odd image dimensions.
        
        x = self.conv5(torch.cat((
            _resize(x, half_size), rgb_half, _resize(logits, half_size),
        ), dim=1))
        x = self.conv6(x)

        full_size = image.shape[-2:]
        original_logits = _resize(logits, full_size)
        
        x = self.conv8(torch.cat((
            _resize(x, full_size), self.conv7(image), original_logits,
        ), dim=1))
        x = self.conv9(x)
        
        return original_logits + self.conv10(x)


