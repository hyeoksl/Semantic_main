import torch.nn as nn
import torch.nn.functional as F


class CustomJSCCLoss(nn.Module):
    def __init__(self, loss_type="MSE"):
        super().__init__()
        if loss_type.upper() != "MSE":
            raise ValueError(f"Unsupported loss type: {loss_type}; expected MSE")

    def forward(self, output, target, training=True):
        mse = F.mse_loss(output["x_hat"], target)
        return {"loss": mse, "MSE": mse}
