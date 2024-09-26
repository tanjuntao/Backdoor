import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, target, other, lambda1=1.0, lambda2=1.0, delta=0.02):
        super(CustomLoss, self).__init__()
        self.target = target
        self.other = other
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = 1.0
        self.delta = delta

    def forward(self, logits):
        probs = torch.softmax(logits, dim=1)

        p_target = probs[:, self.target]
        p_other = probs[:, self.other]

        # L_diff = torch.pow(p_target - p_second, 2)
        L_diff = torch.abs(p_target - p_other)

        mask = torch.ones_like(probs, dtype=bool)
        mask[:, [self.target, self.other]] = False
        p_remains = probs[mask].view(probs.size(0), -1)
        L_max = torch.clamp(
            p_remains + self.delta - p_target.unsqueeze(1), min=0
        ).mean() * p_remains.size(1)

        # loss = self.lambda1 * L_diff.mean() + self.lambda2 * L_max
        loss1 = self.lambda1 * L_diff.mean()
        loss2 = self.lambda2 * L_max
        # loss3 = self.lambda3 * torch.clamp(0.3 - p_other, min=0.0)

        # loss = loss1
        loss = loss1 + loss2
        # loss = loss1 + loss2 + loss3
        return {
            "loss": loss,
            "loss1": loss1,
            "loss2": loss2,
        }
