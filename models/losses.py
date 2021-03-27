import torch


class LabelSmoothedCrossEntropy(torch.nn.Module):

    def __init__(self, ε=0.0, reduction='mean'):
        super().__init__()
        self.ε = ε
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = input.log_softmax(dim=-1)
        weight = input.new_ones(input.size()) * self.ε / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.ε))

        loss = -(weight * log_prob).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class UDACrossEntropy(torch.nn.Module):
    def __init__(self, temperature, threshold, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.reduction = reduction
        self.𝜇ₘₐₛₖ = None

    def forward(self, logits_s, logits_w):
        log_prob = logits_s.log_softmax(dim=-1)
        weight = torch.softmax(logits_w / self.temperature, dim=-1)
        masks = (weight.max(dim=-1)[0] > self.threshold).float()

        loss = -(weight * log_prob).sum(dim=-1) * masks
        self.𝜇ₘₐₛₖ = masks.mean().item()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
