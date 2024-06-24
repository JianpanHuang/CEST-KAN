import torch
import torch.nn as nn

class MSERegLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(MSERegLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        reg_loss = torch.mean(torch.abs(y_pred - y_true))
        loss = mse_loss + self.alpha * reg_loss
        return loss