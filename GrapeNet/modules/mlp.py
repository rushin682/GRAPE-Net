import torch
from torch.nn.init import kaiming_normal_


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=False),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim, bias=False),
            torch.nn.Dropout(dropout)
        )

        # Initialize linear layers with Kaiming normal initialization and batchnorm with constant initialization
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                kaiming_normal_(m.weight)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    pass
