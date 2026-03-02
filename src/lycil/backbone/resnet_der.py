import torch
import torch.nn as nn
import torchvision.models as tvm


def _build_resnet(model="resnet50", *, weights=None, pretrained=False, **kwargs):
    # Max torchvision compatibility: new (weights=) and old (pretrained=)
    builder = getattr(tvm, model) if isinstance(model, str) else model
    try:
        return builder(weights=weights, **kwargs) if weights is not None else builder(pretrained=pretrained, **kwargs)
    except TypeError:
        return builder(pretrained=bool(pretrained or (weights is not None)), **kwargs)


class ResNetWithFeatures(nn.Module):
    """
    Drop-in style wrapper:
    - forward(x): same as torchvision ResNet forward
    - forward_features(x): returns feature_maps + pooled_features
    - optional: cifar_stem, remove_fc
    """
    def __init__(self, model="resnet50", *, weights=None, pretrained=False, cifar_stem=False, remove_fc=True, **kwargs):
        super().__init__()
        self.backbone = _build_resnet(model, weights=weights, pretrained=pretrained, **kwargs)

        if cifar_stem:
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()

        if remove_fc:
            self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

    def forward_features(self, x):
        m = self.backbone
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x1 = m.layer1(x)
        x2 = m.layer2(x1)
        x3 = m.layer3(x2)
        x4 = m.layer4(x3)
        pooled = m.avgpool(x4)
        features = torch.flatten(pooled, 1)
        return {"feature_maps": [x1, x2, x3, x4], "pooled_features": features}


