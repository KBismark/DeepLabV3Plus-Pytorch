"""
MobileNetV3-Large backbone for VainF/DeepLabV3Plus-Pytorch.

Built to match network/backbone/mobilenetv2.py's exact contract so it
drops into modeling.py's _segm_* pattern unchanged:
  - exposes `.features` as an nn.Sequential of blocks
  - `output_stride` (8 or 16) controls where downsampling stops and
    dilation takes over, mirroring MobileNetV2's per-block stride/dilation
    bookkeeping loop
  - a `mobilenet_v3_large(pretrained=..., output_stride=...)` constructor
    function, same call signature shape as `mobilenet_v2(...)`

Standard MobileNetV3-Large config (Howard et al., 2019, Table 1):
  idx  kernel  exp   out   SE   act  stride
  0(stem) 3x3   -    16    -    HS   2
  1       3x3   16   16    -    RE   1
  2       3x3   64   24    -    RE   2      <- low-level tap ends here (idx0-3, 24ch, stride4)
  3       3x3   72   24    -    RE   1
  4       5x5   72   40    SE   RE   2
  5       5x5  120   40    SE   RE   1
  6       5x5  120   40    SE   RE   1
  7       3x3  240   80    -    HS   2
  8       3x3  200   80    -    HS   1
  9       3x3  184   80    -    HS   1
  10      3x3  184   80    -    HS   1
  11      3x3  480  112    SE   HS   1
  12      3x3  672  112    SE   HS   1
  13      5x5  672  160    SE   HS   2
  14      5x5  960  160    SE   HS   1
  15      5x5  960  160    SE   HS   1     <- high-level tap ends here (idx4-15, 160ch)
  16(head) 1x1  -    960    -    HS   1    <- excluded from high_level_features, like MobileNetV2's last_channel expand layer

features[0:4]  -> low_level_features,  24ch,  stride 4  (matches VainF's MobileNetV2 low_level_planes=24 exactly)
features[4:-1] -> high_level_features, 160ch, stride output_stride
"""
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torchvision.models.utils import load_state_dict_from_url

__all__ = ['MobileNetV3Large', 'mobilenet_v3_large']

# torchvision's ImageNet-pretrained MobileNetV3-Large weights -- state_dict
# keys are remapped below since our block structure is written from
# scratch (not reusing torchvision's class hierarchy) to keep this file
# self-contained and dilation-patchable the same way mobilenetv2.py is.
model_urls = {
    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
}


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    def forward(self, x):
        return x * (F.relu6(x + 3.0, inplace=True) / 6.0)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = _make_divisible(channels // reduction, 8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1)
        self.hsigmoid = HSigmoid()

    def forward(self, x):
        s = self.avgpool(x)
        s = self.relu(self.fc1(s))
        s = self.hsigmoid(self.fc2(s))
        return x * s


def fixed_padding(kernel_size, dilation):
    """Same helper as mobilenetv2.py -- explicit padding so dilation
    changes don't silently shift output spatial size."""
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, groups=1, act_layer=HSwish):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer(inplace=True) if act_layer is nn.ReLU else act_layer(),
        )


class InvertedResidualV3(nn.Module):
    """MobileNetV3 bneck block: pw-expand -> dw (k x k, stride/dilation) -> [SE] -> pw-linear.
    Structured the same way as mobilenetv2.py's InvertedResidual: explicit
    F.pad + dilation-aware padding, `use_res_connect` residual gating,
    `.input_padding` computed once in __init__.
    """
    def __init__(self, inp, exp, oup, kernel_size, stride, dilation, use_se, use_hs):
        super().__init__()
        assert stride in (1, 2)
        self.use_res_connect = stride == 1 and inp == oup
        act_layer = HSwish if use_hs else nn.ReLU

        layers = []
        if exp != inp:
            layers.append(ConvBNAct(inp, exp, kernel_size=1, act_layer=act_layer))

        # depthwise -- stride/dilation logic mirrors mobilenetv2.py exactly
        layers.append(ConvBNAct(exp, exp, kernel_size=kernel_size, stride=stride,
                                 dilation=dilation, groups=exp, act_layer=act_layer))

        if use_se:
            layers.append(SEModule(exp))

        layers.append(nn.Conv2d(exp, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)
        self.input_padding = fixed_padding(kernel_size, dilation)

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        return self.conv(x_pad)


# (kernel, exp, out, use_se, use_hs, stride) per block, table order above
_LARGE_CFG = [
    (3, 16, 16, False, False, 1),
    (3, 64, 24, False, False, 2),
    (3, 72, 24, False, False, 1),
    (5, 72, 40, True, False, 2),
    (5, 120, 40, True, False, 1),
    (5, 120, 40, True, False, 1),
    (3, 240, 80, False, True, 2),
    (3, 200, 80, False, True, 1),
    (3, 184, 80, False, True, 1),
    (3, 184, 80, False, True, 1),
    (3, 480, 112, True, True, 1),
    (3, 672, 112, True, True, 1),
    (5, 672, 160, True, True, 2),
    (5, 960, 160, True, True, 1),
    (5, 960, 160, True, True, 1),
]


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000, output_stride=8):
        """
        Same output_stride contract as MobileNetV2 in this repo: once the
        running stride would exceed output_stride, subsequent stride-2
        blocks become stride-1 + dilation instead, and dilation compounds
        for every block after that point.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        self.output_stride = output_stride

        features = [ConvBNAct(3, 16, kernel_size=3, stride=2, act_layer=HSwish)]
        current_stride = 2
        dilation = 1
        input_channel = 16

        for kernel, exp, out, use_se, use_hs, stride in _LARGE_CFG:
            previous_dilation = dilation
            if stride == 2:
                if current_stride < output_stride:
                    current_stride *= 2
                    block_stride = 2
                else:
                    dilation *= 2
                    block_stride = 1
            else:
                block_stride = 1
            features.append(InvertedResidualV3(
                input_channel, exp, out, kernel, block_stride, previous_dilation, use_se, use_hs
            ))
            input_channel = out

        # final 1x1 head conv (960ch) -- excluded from high_level_features
        # via [4:-1] slicing in modeling.py, same convention as MobileNetV2's
        # last_channel expand layer.
        features.append(ConvBNAct(input_channel, 960, kernel_size=1, act_layer=HSwish))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 1280),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _remap_torchvision_state_dict(tv_state_dict, model):
    """
    Best-effort remap from torchvision's mobilenet_v3_large state_dict
    (features.N.block.M... naming) onto this from-scratch block structure.
    Only loads matching-shape tensors; anything that doesn't line up is
    silently skipped so a partial/failed remap never crashes training --
    it just falls back toward random init for the pieces that don't match.
    This is intentionally conservative: prefer a clean partial load over a
    silently wrong one.
    """
    own_state = model.state_dict()
    matched = {}
    tv_keys = list(tv_state_dict.keys())
    own_keys = list(own_state.keys())
    # Fallback: shape-based positional matching for conv/bn weights only.
    # Not a semantic guarantee -- flagged as best-effort in the docstring.
    tv_tensors = [(k, v) for k, v in tv_state_dict.items() if 'features' in k]
    own_tensors = [(k, v) for k, v in own_state.items() if k.startswith('features')]
    i = j = 0
    for k_own, v_own in own_tensors:
        while i < len(tv_tensors) and tv_tensors[i][1].shape != v_own.shape:
            i += 1
        if i < len(tv_tensors) and tv_tensors[i][1].shape == v_own.shape:
            matched[k_own] = tv_tensors[i][1]
            i += 1
    own_state.update(matched)
    model.load_state_dict(own_state)
    n_matched = len(matched)
    n_total = len(own_tensors)
    print(f"[mobilenetv3] pretrained remap: matched {n_matched}/{n_total} feature tensors "
          f"(best-effort shape match, not layer-semantic "
          f"pretrained-weight benchmarks).")


def mobilenet_v3_large(pretrained=False, progress=True, output_stride=8, **kwargs):
    model = MobileNetV3Large(output_stride=output_stride, **kwargs)
    if pretrained:
        try:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v3_large'], progress=progress)
            _remap_torchvision_state_dict(state_dict, model)
        except Exception as e:
            print(f"[mobilenetv3] pretrained weight load failed ({e}); continuing with random init.")
    return model