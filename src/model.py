"""Model builder. Paper architecture (Fig. 1):
ResNet18 backbone -> AvgPool -> FC -> Dropout -> ReLU -> Dropout -> ReLU -> OSL -> softmax
Other backbones / the plain FC head are included so you can reproduce the paper's comparison tables."""
import torch.nn as nn
from torchvision import models
from .osl import OrthogonalSoftmaxLayer

BACKBONES = ["resnet18", "resnet50", "mobilenet_v2", "vgg16"]


def _make_head(in_dim, num_classes, head, hidden, p):
    if head == "fc":      # baseline: ordinary fully connected classification layer
        return nn.Linear(in_dim, num_classes)
    if head == "osl":     # proposed
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Dropout(p), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.ReLU(inplace=True),
            OrthogonalSoftmaxLayer(hidden, num_classes),
        )
    raise ValueError(head)


def build_model(name="resnet18", head="osl", num_classes=2, hidden=256, dropout=0.5, pretrained=True):
    if name not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}")
    w = "DEFAULT" if pretrained else None
    if name in ("resnet18", "resnet50"):
        m = getattr(models, name)(weights=w)
        m.fc = _make_head(m.fc.in_features, num_classes, head, hidden, dropout)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=w)
        m.classifier = _make_head(m.last_channel, num_classes, head, hidden, dropout)
    else:  # vgg16
        m = models.vgg16(weights=w)
        m.classifier = _make_head(512 * 7 * 7, num_classes, head, hidden, dropout)
    return m
