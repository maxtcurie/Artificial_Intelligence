from torch import nn
import torch

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target):
        # Create a mask where the target is not NaN
        mask = ~torch.isnan(target)

        # Calculate the mean over the masked elements
        masked_input = input[mask]
        masked_target = target[mask]

        # Calculate the MSE only for the masked elements
        loss = nn.functional.mse_loss(masked_input, masked_target, reduction = 'sum')
        
        return loss


class MaskedBCELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(MaskedBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # Ensure input and target have the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape for masked BCE loss.")

        # Create a mask where the target is not NaN
        mask = ~torch.isnan(target)
        
        # Apply mask to input and target
        masked_input = input[mask]
        masked_target = target[mask]

        # Compute BCE only for the masked elements
        loss = nn.functional.binary_cross_entropy(masked_input, masked_target, reduction=self.reduction)
        
        return loss
