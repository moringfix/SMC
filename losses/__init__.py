from torch import nn 
from .SupConLoss import SupConLoss, SupConLossWeighted

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'SupConLoss': SupConLoss(),
                'MSELoss': nn.MSELoss(),
                'SupConLossWeighted': SupConLossWeighted(),
            }
