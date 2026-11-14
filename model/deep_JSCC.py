import torch
import torch.nn as nn
import os, yaml 
from einops import rearrange
def conv(in_ch, out_ch, kernel_size=5, stride=2):
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_ch, out_ch, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, output_padding=stride - 1)

from model.layers import *

__all__ = ['DeepJSCC_lite']

class DeepJSCC_lite(nn.Module):
    def __init__(self, spp=0.25, dim=384, **kwargs):
        self.spp=spp
        super().__init__()
        downscale_factor = 2 ** 4
        symbol_dim = 2 * int(spp * downscale_factor**2)
        if symbol_dim < 2 or dim <= 0 or dim % 2:
            raise ValueError('spp must allow at least one symbol and dim must be a positive even integer')
        
        self.encoder = nn.Sequential(
            conv(3,dim,3),
            resNetBlock(dim,dim),
            conv(dim,3*dim//2,3),
            resNetBlock(3*dim//2,3*dim//2),
            conv(3*dim//2,2*dim,3),
            resNetBlock(2*dim,2*dim),
            conv(2*dim,3*dim,3),
            resNetBlock(3*dim,3*dim),
            nn.Conv2d(3*dim, symbol_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(symbol_dim, 3*dim, 1),
            resNetBlock(3*dim,3*dim),
            deconv(3*dim,2*dim,3),
            resNetBlock(2*dim,2*dim),
            deconv(2*dim,3*dim//2,3),
            resNetBlock(3*dim//2,3*dim//2),
            deconv(3*dim//2,dim,3),
            resNetBlock(dim,dim),
            deconv(dim,3,3),
        )
        self.channel = AWGNChannel
    
    def get_parser(parser):
        parser.add_argument('--model.min_SNRdB', type=float)
        parser.add_argument('--model.max_SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.min_spp', type=float)
        parser.add_argument('--model.dim', type=int)
        
        return parser
    
    def forward(self, x:torch.Tensor, snr:float, spp:float):
        y = self.encoder(x) # (B, symbol_dim=128, 16, 16)
        B,C,H,W=y.shape
        if not 0 < spp <= self.spp:
            raise ValueError('spp must be positive and no greater than the configured maximum')
        rem_chan = max(2, min(C, 2 * round(spp / self.spp * (C // 2))))
        z=rearrange(y[:,:rem_chan],"b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        z=torch.complex(z[...,0], z[..., 1])
        
        z_hat = self.channel(z, snr)

        z_hat = torch.stack((z_hat.real, z_hat.imag), dim=-1)
        y_hat = rearrange(z_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W)
        y_hat_full=y.new_zeros(B,C,H,W)
        y_hat_full[:,:rem_chan]=y_hat
        
        x_hat = self.decoder(y_hat_full) # (B, 3, 256, 256)
        x_hat = torch.clamp(x_hat, 0, 1)

        # return x_hat
        return {
            "x_hat": x_hat, "y":y
        }

