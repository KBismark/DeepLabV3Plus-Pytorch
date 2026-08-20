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
            if aux_in_channels is None:
                raise ValueError(
                    "aux_in_channels must be provided when use_aux=True "
                    "(eager construction needs the channel count upfront -- "
                    "see main_geo.py for how to infer it from base_model.backbone)."
                )
            self.prior_head = ConvHead(aux_in_channels, num_classes, aux_hidden_ch)
            self.boundary_head = ConvHead(aux_in_channels, 1, aux_hidden_ch)
        else:
            self.prior_head = None
            self.boundary_head = None

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
            prior = torch.tanh(self.prior_head(backbone_feature))
            boundary = torch.sigmoid(self.boundary_head(backbone_feature))
            out["prior"] = F.interpolate(prior, size=input_shape, mode='bilinear', align_corners=False)
            out["boundary"] = F.interpolate(boundary, size=input_shape, mode='bilinear', align_corners=False)

        return out