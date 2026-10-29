"""Orthogonal SoftMax Layer (OSL) - the classifier head from the paper (after OSLNet, Li et al. 2020).

A normal fully-connected classification layer (FCCL) is  logits = W x + b  with a dense W (n_classes x D).
OSL multiplies W by a FIXED binary mask M (n_classes x D) in which every input feature is wired to exactly
one class. So the weight vectors of different classes use disjoint features => they are pairwise orthogonal
(the inter-class angle is 90 degrees). The mask never changes, in training or in testing (unlike DropConnect).
The softmax itself is applied by the loss (CrossEntropyLoss) / by torch.softmax at inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalSoftmaxLayer(nn.Module):
    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        if in_features < num_classes:
            raise ValueError("in_features must be >= num_classes for OSL")
        self.in_features, self.num_classes = in_features, num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.bias = nn.Parameter(torch.zeros(num_classes)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Mask M[j, i] = 1 if feature i belongs to class j (round-robin assignment), else 0.
        mask = torch.zeros(num_classes, in_features)
        mask[torch.arange(in_features) % num_classes, torch.arange(in_features)] = 1.0
        self.register_buffer("mask", mask)  # buffer: saved in state_dict, not trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    @torch.no_grad()
    def max_cosine_between_classes(self) -> float:
        """Should be 0.0 - proves the class weight vectors are orthogonal."""
        w = F.normalize(self.weight * self.mask, dim=1)
        g = w @ w.t()
        g.fill_diagonal_(0)
        return g.abs().max().item()

    def extra_repr(self):
        return f"in_features={self.in_features}, num_classes={self.num_classes}"
