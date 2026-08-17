"""
Geometric supervision wrapper for VainF's DeepLabV3+.

Design mirrors GASNetLite exactly: prior_head/boundary_head consume the
SAME backbone feature tensor that feeds ASPP/decoder ('out', the
high-level feature), but are NOT wired into the classifier's computation
graph. This is what makes them genuinely discardable at inference -- the
segmentation path (backbone -> classifier) is byte-for-byte identical to
stock DeepLabV3+, whether or not this wrapper's aux heads exist.

Usage:
    from network import modeling
    from geo_wrapper import GeoDeepLabV3Plus

    base_model = modeling.deeplabv3plus_mobilenetv3_large(num_classes=21, output_stride=8)
    model = GeoDeepLabV3Plus(base_model, num_classes=21, use_aux=True)   # with geometric supervision
    # or
    model = GeoDeepLabV3Plus(base_model, num_classes=21, use_aux=False)  # clean baseline, no aux heads at all

Both variants share identical backbone + ASPP + decoder code (VainF's
unmodified DeepLabV3+) -- the only difference is whether prior_head/
boundary_head exist and run during training. At eval/inference, aux
heads never run regardless of use_aux (see forward()).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvHead(nn.Module):
    """Same shape as gasnet_lite/heads.py::ConvHead -- reused pattern for consistency."""
    def __init__(self, in_ch, out_ch, hidden_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


class GeoDeepLabV3Plus(nn.Module):
    """
    Wraps a VainF DeepLabV3+ model (from network.modeling) to add optional,
    training-only geometric supervision heads.

    base_model: an already-constructed model from network.modeling, e.g.
                modeling.deeplabv3plus_mobilenetv3_large(...). Its
                .backbone and .classifier are used as-is, completely
                unmodified -- this wrapper only adds new branches, never
                edits the existing DeepLabV3+ computation.
    """

    def __init__(self, base_model, num_classes, use_aux=False, aux_hidden_ch=64,
                 aux_in_channels=None):
        super().__init__()
        self.base_model = base_model     # has .backbone (IntermediateLayerGetter) and .classifier
        self.num_classes = num_classes
        self.use_aux = use_aux

        if use_aux:
            # in_channels must match the backbone's 'out' feature channel
            # count -- 160 for MobileNetV3-Large, 320 for MobileNetV2,
            # 2048 for ResNet50/101, 2048 for Xception. Pass explicitly to
            # avoid guessing wrong for a given backbone; falls back to a
            # lazy first-forward inference if not provided.
            self._aux_in_channels = aux_in_channels
            self.prior_head = None    # built lazily on first forward if aux_in_channels is None
            self.boundary_head = None
            self._aux_hidden_ch = aux_hidden_ch
        else:
            self.prior_head = None
            self.boundary_head = None

    def _build_aux_heads(self, in_channels, device):
        self.prior_head = ConvHead(in_channels, self.num_classes, self._aux_hidden_ch).to(device)
        self.boundary_head = ConvHead(in_channels, 1, self._aux_hidden_ch).to(device)

    def forward(self, x, run_aux=None):
        if run_aux is None:
            run_aux = self.use_aux and self.training

        input_shape = x.shape[-2:]
        features = self.base_model.backbone(x)          # dict: {'out': ..., 'low_level': ...}
        logits = self.base_model.classifier(features)    # unmodified DeepLabV3+ decoder path
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)

        mask = torch.sigmoid(logits) if self.num_classes == 1 else torch.softmax(logits, dim=1)
        out = {"logits": logits, "mask": mask}

        if run_aux:
            backbone_feature = features['out']            # same tensor ASPP consumes -- untouched, just read
            if self.prior_head is None:
                in_ch = self._aux_in_channels or backbone_feature.shape[1]
                self._build_aux_heads(in_ch, backbone_feature.device)

            prior = torch.tanh(self.prior_head(backbone_feature))
            boundary = torch.sigmoid(self.boundary_head(backbone_feature))
            out["prior"] = F.interpolate(prior, size=input_shape, mode='bilinear', align_corners=False)
            out["boundary"] = F.interpolate(boundary, size=input_shape, mode='bilinear', align_corners=False)

        return out