import torch
import torch.nn as nn

# ASDLoss inherits from nn.Module to define a custom loss function
class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        # Initialize the CrossEntropyLoss function
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # Compute the cross-entropy loss
        loss = self.ce(logits, labels)
        return loss
