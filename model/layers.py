import torch
import torch.nn as nn
import torch.nn.functional as F
def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
from einops import rearrange

__all__ = [
    "resNetBlock", "AFModule", "AWGNChannel",
]

def AWGNChannel(x: torch.Tensor, snr:float) -> torch.Tensor:        
    # Noise Generation
    sig_power = torch.mean(torch.abs(x)**2)
    no = torch.sqrt(sig_power/(10 ** (snr / 10)))
    n = no * torch.randn(x.shape, dtype=torch.complex64, device=x.device)
    # Add noise
    y_hat = x + n
    
    return y_hat

    
class resNetBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """
    def __init__(self, in_ch: int,out_ch:int):
        super().__init__()
        self.path=nn.Sequential(
            conv1x1(in_ch, in_ch//2),
            nn.ReLU(inplace=True),
            conv3x3(in_ch//2, in_ch//2),
            nn.ReLU(inplace=True),
            conv1x1(in_ch//2, out_ch)
        )
        self.skip=conv1x1(in_ch, out_ch) if in_ch!=out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)+self.skip(x)
    
class AFModule(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        hidden_dim=max(in_ch//8, 8)
        self.predictor=nn.Sequential(
            nn.Linear(in_ch+1, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_ch, bias=True),
            nn.Sigmoid()
        )
    def forward(self,x: torch.Tensor, snr: float):
        b, c, _, _=x.shape
        pooled_features=x.mean(dim=[2, 3])
        pooled_features=pooled_features.view(b, c)
        snr_feature=torch.full((b, 1), snr, device=x.device, dtype=x.dtype)
        pooled_features=torch.cat([snr_feature, pooled_features], dim=1)
        scale_factors=self.predictor(pooled_features)
        scale_features=scale_factors.view(b,c,1,1)
        re=x*scale_features
        return re
