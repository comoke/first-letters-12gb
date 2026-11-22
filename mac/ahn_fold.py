"""ahn_fold.py : the winner's A-hn network (ink8um.InkDetector: v8-in ResNet3D-50 encoder + 2-D decoder) rewritten with 2-D ops only.

Same weights, same arithmetic up to float rounding; built from a loaded InkDetector by fold(model). Why: Apple's fast paths (the
Neural Engine through Core ML, MPS's native 2-D convolutions, Metal's matrix units) are 2-D. PyTorch-MPS runs nn.Conv3d as one large
matrix product, near CPU speed (villa PR #2634), and the Neural Engine has no 3-D convolution.

Layout: a tile's activations live as (D, C, H, W), depth as the batch axis, so every op is 4-D (no rank-5 tensor anywhere, which Core
ML's Neural Engine path prefers) and no permute is needed between layers. A batch of B tiles is B such blocks stacked: (B*D, C, H, W).

What changes, layer by layer:
  * every BatchNorm is folded into the convolution before it (inference statistics; computed in float64, then cast);
  * InstanceNorm3d(1, affine) is written out (mean and biased variance over the tile, eps 1e-5, then the affine);
  * the input's trilinear upsample (24, 64, 64) -> (96, 256, 256) is split into its two separable parts: a fixed (96 x 24) linear map
    over depth, then a bilinear 2-D upsample (align_corners=False: trilinear's source-index rule, axis by axis);
  * the stem's Conv3d(1 -> 64, k 7x7x7, stride (1,2,2), pad 3) becomes one Conv2d whose 7 input channels are the 7 depth taps;
  * MaxPool3d(1,3,3) is MaxPool2d(3);
  * a 1x1x1 Conv3d is a 1x1 Conv2d; with stride 2 it keeps every other depth plane and strides 2 in H, W;
  * a 3x3x3 Conv3d (stride 1 or 2, zero pad 1) is either
      'stack': one Conv2d over the 3 depth-shifted copies stacked as channels (weight (out, 3C, 3, 3)), as HerculeX's
               stock_model.folded does for its stem, or
      'sum'  : (stride 1 only) three Conv2d over the unshifted input, one per depth tap, added with a depth shift
               (no 3x input copy; #2634's form for large layers);
  * the max over depth that feeds the decoder is an amax over the depth axis;
  * the 2-D decoder is unchanged except for its folded BatchNorms.

Inputs and outputs are InkDetector's: x (B, 24, 64, 64) float (uint8 clipped at 200, / 255) -> logits (B, 1, 64, 64).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fold_bn(weight, bn):
    """conv weight (out, ...) and the BatchNorm after it -> (weight, bias) of one conv, in float64."""
    w = weight.detach().double()
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
    bias = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    return w * scale.reshape(-1, *([1] * (w.ndim - 1))), bias


def _depth_matrix(d_in, d_out):
    """(d_out, d_in) matrix of 1-D linear interpolation, align_corners=False (trilinear's rule along depth)."""
    eye = torch.eye(d_in, dtype=torch.float64)[:, None, :]
    return F.interpolate(eye, size=d_out, mode='linear', align_corners=False)[:, 0, :].T.contiguous()


# Every size below is a Python int carried alongside the tensor (b tiles, d planes each), never read from a tensor's shape, so a
# traced graph has only constants in it (Core ML's converter rejects shape arithmetic).

def _per_tile(x, b, d, fn):
    """Apply fn to each tile's (d, C, H, W) block of a (b*d, C, H, W) tensor and stack the results back."""
    if b == 1:
        return fn(x)
    return torch.cat([fn(x[i * d:(i + 1) * d]) for i in range(b)], 0)


def _pad_depth(x, before, after):
    """Zero planes before and after a (d, C, H, W) block (a constant pad of axis 0)."""
    return F.pad(x, (0, 0, 0, 0, 0, 0, before, after))


class Conv3x3x3(nn.Module):
    """A 3x3x3 Conv3d (zero pad 1, stride s in all three axes) with a folded BatchNorm as bias."""

    def __init__(self, weight, bias, stride, mode):
        super().__init__()
        out, cin = weight.shape[:2]
        self.stride, self.mode = stride, mode if stride == 1 else 'stack'
        if self.mode == 'stack':
            self.register_buffer('w', weight.permute(0, 2, 1, 3, 4).reshape(out, 3 * cin, 3, 3).contiguous())
        else:
            self.register_buffer('w', weight.permute(2, 0, 1, 3, 4).contiguous())     # (3 taps, out, cin, 3, 3)
        self.register_buffer('bias', bias.contiguous())

    def forward(self, x, b, d):
        """-> (output, its depth)"""
        s = self.stride
        if self.mode == 'sum':                                  # out[d] = W0 x[d-1] + W1 x[d] + W2 x[d+1], zeros past the ends
            y = [F.conv2d(x, self.w[k], None, 1, 1) for k in range(3)]

            def add(i):
                a0, a1, a2 = (t[i * d:(i + 1) * d] for t in y)
                return a1 + _pad_depth(a0[:d - 1], 1, 0) + _pad_depth(a2[1:], 0, 1)
            out = add(0) if b == 1 else torch.cat([add(i) for i in range(b)], 0)
            return out + self.bias.reshape(1, -1, 1, 1), d
        do = (d - 1) // s + 1

        def one(t):
            tp = _pad_depth(t, 1, 1)
            return torch.cat([tp[k:k + s * (do - 1) + 1:s] for k in range(3)], 1)         # (do, 3C, H, W)
        return F.conv2d(_per_tile(x, b, d, one), self.w, self.bias, s, 1), do


class Conv1x1x1(nn.Module):
    def __init__(self, weight, bias, stride):
        super().__init__()
        self.stride = stride
        self.register_buffer('w', weight.reshape(*weight.shape[:2], 1, 1).contiguous())
        self.register_buffer('bias', bias.contiguous())

    def forward(self, x, b, d):
        """-> (output, its depth)"""
        if self.stride != 1:                                                            # every other depth plane; stride 2 in H, W
            x = _per_tile(x, b, d, lambda t: t[::self.stride])
            d = (d - 1) // self.stride + 1
        return F.conv2d(x, self.w, self.bias, self.stride), d


class Bottleneck(nn.Module):
    def __init__(self, blk, mode):
        super().__init__()
        self.c1 = Conv1x1x1(*_fold_bn(blk.conv1.weight, blk.bn1), 1)
        self.c2 = Conv3x3x3(*_fold_bn(blk.conv2.weight, blk.bn2), blk.conv2.stride[0], mode)
        self.c3 = Conv1x1x1(*_fold_bn(blk.conv3.weight, blk.bn3), 1)
        self.down = None
        if blk.downsample is not None:
            conv, bn = blk.downsample[0], blk.downsample[1]
            self.down = Conv1x1x1(*_fold_bn(conv.weight, bn), conv.stride[0])

    def forward(self, x, b, d):
        out, _ = self.c1(x, b, d)
        out, do = self.c2(F.relu(out), b, d)
        out, _ = self.c3(F.relu(out), b, do)
        res = self.down(x, b, d)[0] if self.down is not None else x
        return F.relu(out + res), do


class FoldedAhn(nn.Module):
    def __init__(self, model, mode='stack', dtype=torch.float32):
        super().__init__()
        bb = model.backbone
        self.depth_in, self.depth_up, self.hw_up = model.in_depth, model.input_upsample_depth, model.input_upsample_hw
        nrm = model.normalization
        self.norm_eps = float(nrm.eps)
        self.register_buffer('norm_w', nrm.weight.detach().double().reshape(1, 1, 1, 1).clone())
        self.register_buffer('norm_b', nrm.bias.detach().double().reshape(1, 1, 1, 1).clone())
        self.register_buffer('depth_matrix', _depth_matrix(self.depth_in, self.depth_up))
        w, bias = _fold_bn(bb.conv1.weight, bb.bn1)                                     # (64, 1, 7, 7, 7)
        self.register_buffer('stem_w', w[:, 0].contiguous())                           # (64, 7 depth taps, 7, 7)
        self.register_buffer('stem_b', bias)
        self.stem_taps = bb.conv1.kernel_size[0]
        self.layers = nn.ModuleList([nn.ModuleList([Bottleneck(blk, mode) for blk in layer])
                                     for layer in (bb.layer1, bb.layer2, bb.layer3, bb.layer4)])
        self.dec_w, self.dec_b = [], []
        for i, seq in enumerate(model.decoder.convs):
            cw, cb = _fold_bn(seq[0].weight, seq[1])
            self.register_buffer(f'dec_w{i}', cw); self.register_buffer(f'dec_b{i}', cb)
        self.register_buffer('logit_w', model.decoder.logit.weight.detach().double().clone())
        self.register_buffer('logit_b', model.decoder.logit.bias.detach().double().clone())
        self.n_dec = len(model.decoder.convs)
        self.tile = model.tile_size
        self.batch = None                       # set to a fixed batch for tracing (export): then no size is read from a tensor
        self.to(dtype)

    def forward(self, x):
        if x.ndim == 5:
            x = x[:, 0]
        b, h, w = self.batch or x.shape[0], self.tile, self.tile
        mean = x.mean((1, 2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean((1, 2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.norm_eps) * self.norm_w + self.norm_b         # InstanceNorm3d(1, affine)
        x = torch.matmul(self.depth_matrix, x.reshape(b, self.depth_in, h * w)).reshape(b, self.depth_up, h, w)
        x = F.interpolate(x, size=(self.hw_up, self.hw_up), mode='bilinear', align_corners=False)   # (b, 96, 256, 256)
        p, t = self.stem_taps // 2, self.stem_taps

        def stem_in(i):
            xp = _pad_depth(x[i].reshape(self.depth_up, 1, self.hw_up, self.hw_up), p, p)
            return torch.cat([xp[k:k + self.depth_up] for k in range(t)], 1)            # (96, 7, 256, 256)
        stacked = stem_in(0) if b == 1 else torch.cat([stem_in(i) for i in range(b)], 0)
        x = F.max_pool2d(F.relu(F.conv2d(stacked, self.stem_w, self.stem_b, 2, p)), 3, 2, 1)   # (b*96, 64, 64, 64)
        pooled, d = [], self.depth_up
        for layer in self.layers:
            for blk in layer:
                x, d = blk(x, b, d)
            pooled.append(torch.cat([x[i * d:(i + 1) * d].amax(0, keepdim=True) for i in range(b)], 0) if b > 1
                          else x.amax(0, keepdim=True))
        for i in range(len(pooled) - 1, 0, -1):
            up = F.interpolate(pooled[i], scale_factor=2, mode='bilinear')
            f = torch.cat([pooled[i - 1], up], 1)
            pooled[i - 1] = F.relu(F.conv2d(f, getattr(self, f'dec_w{i - 1}'), getattr(self, f'dec_b{i - 1}'), 1, 1))
        return F.conv2d(pooled[0], self.logit_w, self.logit_b)


def fold(model, mode='stack', dtype=torch.float32):
    """A loaded InkDetector in eval mode -> FoldedAhn (eval) whose buffers are cast to dtype after folding in float64."""
    assert not model.training, 'fold an eval-mode model (BatchNorm uses running statistics)'
    return FoldedAhn(model, mode, dtype).eval()
