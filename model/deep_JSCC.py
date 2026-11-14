import torch
import torch.nn as nn

from model.layers import resNetBlock

__all__ = ['DeepJSCCEncoder', 'DeepJSCCDecoder']


def conv(in_ch, out_ch, kernel_size=5, stride=2):
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_ch, out_ch, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, output_padding=stride - 1)


def symbol_channels(spp, dim):
    channels = 2 * int(spp * 256)
    if channels < 2 or dim <= 0 or dim % 2:
        raise ValueError('spp must allow at least one symbol and dim must be a positive even integer')
    return channels


class DeepJSCCEncoder(nn.Sequential):
    """Encode images as interleaved real/imaginary symbol channels."""

    def __init__(self, spp=0.25, dim=384):
        channels = symbol_channels(spp, dim)
        super().__init__(
            conv(3, dim, 3), resNetBlock(dim, dim),
            conv(dim, 3 * dim // 2, 3), resNetBlock(3 * dim // 2, 3 * dim // 2),
            conv(3 * dim // 2, 2 * dim, 3), resNetBlock(2 * dim, 2 * dim),
            conv(2 * dim, 3 * dim, 3), resNetBlock(3 * dim, 3 * dim),
            nn.Conv2d(3 * dim, channels, 1),
        )
        self.max_spp = spp
        self.channels = channels

    def forward(self, image, spp):
        if not 0 < spp <= self.max_spp:
            raise ValueError('spp must be positive and no greater than the configured maximum')
        active = max(2, min(self.channels, 2 * round(spp / self.max_spp * (self.channels // 2))))
        return super().forward(image)[:, :active]


class DeepJSCCDecoder(nn.Sequential):
    """Decode received I/Q channels, cropping to the original image size."""

    def __init__(self, spp=0.25, dim=384):
        channels = symbol_channels(spp, dim)
        super().__init__(
            nn.Conv2d(channels, 3 * dim, 1), resNetBlock(3 * dim, 3 * dim),
            deconv(3 * dim, 2 * dim, 3), resNetBlock(2 * dim, 2 * dim),
            deconv(2 * dim, 3 * dim // 2, 3), resNetBlock(3 * dim // 2, 3 * dim // 2),
            deconv(3 * dim // 2, dim, 3), resNetBlock(dim, dim), deconv(dim, 3, 3),
        )
        self.channels = channels

    def forward(self, received, height, width):
        padding = received.new_zeros(received.shape[0], self.channels - received.shape[1],
                                     received.shape[2], received.shape[3])
        full = torch.cat((received, padding), dim=1)
        return super().forward(full).clamp(0, 1)[:, :, :height, :width]
