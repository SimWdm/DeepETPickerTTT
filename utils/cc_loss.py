from torch import nn


class CCLoss3D(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super(CCLoss3D, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def _normalize(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + self.eps)

    def forward(self, x, y):
        if x.ndim != 5 or y.ndim != 5:
            raise ValueError("Input tensors must be 5-dimensional (B, C, D, H, W)")
        if x.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError("Input tensors must have a single channel (C=1) for now!")
        # flatten x along self.dims, keep other dimensions
        x_flat = x.squeeze(1).flatten(start_dim=1, end_dim=3)
        y_flat = y.squeeze(1).flatten(start_dim=1, end_dim=3)
        x_norm = self._normalize(x_flat)
        y_norm = self._normalize(y_flat)
        cc = 1 - (x_norm * y_norm).sum(dim=-1)
        if self.reduction == 'mean':
            return cc.mean()   
        elif self.reduction == 'sum':
            return cc.sum()
        elif self.reduction == 'none' or self.reduction is None:
            return cc
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}. Use 'mean', 'sum', or 'none'.")
