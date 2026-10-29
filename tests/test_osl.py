import torch
from src.osl import OrthogonalSoftmaxLayer
from src.model import build_model


def test_orthogonal():
    l = OrthogonalSoftmaxLayer(256, 2)
    assert l.max_cosine_between_classes() < 1e-7
    x = torch.randn(4, 256)
    assert l(x).shape == (4, 2)
    l(x).sum().backward()
    # masked-out weights must receive zero gradient (they are never used)
    assert (l.weight.grad * (1 - l.mask)).abs().sum() == 0


def test_model_shapes():
    for head in ["fc", "osl"]:
        m = build_model("resnet18", head, pretrained=False).eval()
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 2)


if __name__ == "__main__":
    test_orthogonal(); test_model_shapes(); print("all tests passed")
