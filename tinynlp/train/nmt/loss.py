import torch


class Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, mask):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1).long()
        mask = mask.view(-1)
        outputs = outputs[mask]
        targets = targets[mask]
        loss = self.fn(outputs, targets)
        return loss
