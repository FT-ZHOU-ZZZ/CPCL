import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MyLoss(nn.Module):
    def __init__(self, margin, beta):
        super(MyLoss, self).__init__()
        print(margin)
        print(beta)
        self.margin = margin
        self.beta = beta

    def forward(self, gt_label, pre_label, decoupling_distance, coupling_distance):
        classification_loss = F.multilabel_soft_margin_loss(pre_label, gt_label)
        metric_loss = triplet_loss(gt_label, decoupling_distance, coupling_distance, self.margin)
        return classification_loss + self.beta * metric_loss


def triplet_loss(gt_label, decoupling_distance, coupling_distance, margin=20.0):
    decoupling_pairs = torch.mul(torch.clamp(decoupling_distance - coupling_distance + margin, min=0.0), 1 - gt_label)
    coupling_pairs = torch.mul(torch.clamp(coupling_distance - decoupling_distance + margin, min=0.0), gt_label)
    return torch.mean(decoupling_pairs + coupling_pairs)