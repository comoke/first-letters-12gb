"""ahn_variants.py : the winner's A-hn network (ink8um.InkDetector) in the forms the 4070 work measures.

Every builder takes the loaded eval-mode InkDetector and returns a callable  f(x) -> logits  for x = (B, 24, 64, 64) float32 on CUDA
(uint8 clipped at 200, / 255: what infer.band_sums feeds the model).  'stock' is the winner's own call.  Nothing here changes weights
other than by folding BatchNorm (inference statistics) into the convolution before it.

  stock      model(x) under fp16 autocast (infer.py as shipped)
  cl3        the same, weights and activations in channels_last_3d
  fuse       BatchNorm folded (float64, then float32), fp16 autocast, NCDHW
  fuse_cl3   folded + channels_last_3d, fp16 autocast
  half_cl3   folded + channels_last_3d, weights stored in fp16 (no per-call weight cast); the input norm and the trilinear upsample
             stay in float32, as autocast runs them
  half       as half_cl3 in NCDHW
  fold2d     mac/ahn_fold.FoldedAhn ('stack': every 3-D convolution as a 2-D one over depth taps), channels_last, fp16 autocast
A '+c' suffix compiles the module with torch.compile; '+ra' compiles with mode='reduce-overhead' (CUDA graphs).
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fused(model):
    """A deep copy with every BatchNorm folded into its convolution (computed in float64)."""
    m = copy.deepcopy(model).double().eval()
    bb = m.backbone
    bb.conv1 = fuse_conv_bn_eval(bb.conv1, bb.bn1); bb.bn1 = nn.Identity()
    for layer in (bb.layer1, bb.layer2, bb.layer3, bb.layer4):
        for blk in layer:
            blk.conv1 = fuse_conv_bn_eval(blk.conv1, blk.bn1); blk.bn1 = nn.Identity()
            blk.conv2 = fuse_conv_bn_eval(blk.conv2, blk.bn2); blk.bn2 = nn.Identity()
            blk.conv3 = fuse_conv_bn_eval(blk.conv3, blk.bn3); blk.bn3 = nn.Identity()
            if blk.downsample is not None:
                blk.downsample = nn.Sequential(fuse_conv_bn_eval(blk.downsample[0], blk.downsample[1]))
    for seq in m.decoder.convs:
        seq[0] = fuse_conv_bn_eval(seq[0], seq[1]); seq[1] = nn.Identity()
    return m.float().eval()


def cl3(m):
    """The 3-D encoder's weights in channels_last_3d and the 2-D decoder's in channels_last, in place."""
    m.backbone.to(memory_format=torch.channels_last_3d); m.decoder.to(memory_format=torch.channels_last)
    return m


class Split(nn.Module):
    """InkDetector with the input norm and upsample in float32 and the encoder and decoder in the weights' own dtype and layout."""

    def __init__(self, model, cl3):
        super().__init__()
        self.m, self.cl3 = model, cl3
        self.size = (model.input_upsample_depth, model.input_upsample_hw, model.input_upsample_hw)

    def forward(self, x):
        m = self.m
        x = F.interpolate(m.normalization(x[:, None]), size=self.size, mode='trilinear', align_corners=False)
        x = x.to(m.backbone.conv1.weight.dtype)
        if self.cl3:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        pooled = [torch.max(f, dim=2)[0] for f in m.backbone(x)]
        return m.decoder(pooled)


class CuFused(nn.Module):
    """InkDetector with every convolution's bias, ReLU and residual add done inside cuDNN's fused calls.

    Why: on this network the convolutions are about 40 % of GPU time; the rest is separate passes over activations of up to
    200 MB per tile (BatchNorm or bias, ReLU, the residual add) and layout conversions.  Folding BatchNorm alone gains nothing,
    because PyTorch then adds the bias as a pass of its own.  torch.cudnn_convolution_relu and torch.cudnn_convolution_add_relu
    hand conv + bias + ReLU (+ the residual) to cuDNN as one operation.
      * BatchNorm is folded first (fused()); weights are fp16, activations channels-last when cl.
      * a 1x1x1 convolution runs as a 2-D 1x1 one on the (B, C, D*H, W) view of the volume (a view, no copy); with stride 2 the
        volume is subsampled first, as the 3-D convolution does.
      * the block's last convolution takes the residual as cuDNN's z: relu(conv(x) + bias + residual) in one call.
      * the 3x3x3 convolution goes through the same fused call (three=True); where cuDNN has no fused 3-D engine PyTorch's own
        fallback runs the convolution and adds bias and ReLU in place.
    The input norm and the trilinear upsample stay in float32, as autocast runs them.
    """

    def __init__(self, model, cl=True, three=True):
        super().__init__()
        f = fused(model)
        self.norm, self.cl, self.three = f.normalization, cl, three
        self.size = (f.input_upsample_depth, f.input_upsample_hw, f.input_upsample_hw)
        bb = f.backbone
        self.pool = bb.maxpool
        self.stem_stride, self.stem_pad = tuple(bb.conv1.stride), tuple(bb.conv1.padding)
        self.register_buffer('stem_w', self._w5(bb.conv1.weight)); self.register_buffer('stem_b', bb.conv1.bias.detach().half())
        self.blocks = []
        for li, layer in enumerate((bb.layer1, bb.layer2, bb.layer3, bb.layer4)):
            for bi, blk in enumerate(layer):
                k = f'l{li}b{bi}'
                self.register_buffer(k + 'w1', self._w4(blk.conv1.weight)); self.register_buffer(k + 'b1', blk.conv1.bias.detach().half())
                self.register_buffer(k + 'w2', self._w5(blk.conv2.weight)); self.register_buffer(k + 'b2', blk.conv2.bias.detach().half())
                self.register_buffer(k + 'w3', self._w4(blk.conv3.weight)); self.register_buffer(k + 'b3', blk.conv3.bias.detach().half())
                down = blk.downsample is not None
                if down:
                    self.register_buffer(k + 'wd', self._w4(blk.downsample[0].weight)); self.register_buffer(k + 'bd', blk.downsample[0].bias.detach().half())
                self.blocks.append((k, int(blk.conv2.stride[0]), down, int(blk.downsample[0].stride[0]) if down else 1, bi == len(layer) - 1))
        self.decoder = f.decoder.half()
        if cl: self.decoder.to(memory_format=torch.channels_last)

    def _w4(self, w):
        w = w.detach().half().reshape(w.shape[0], w.shape[1], 1, 1)
        return w.contiguous(memory_format=torch.channels_last) if self.cl else w.contiguous()

    def _w5(self, w):
        w = w.detach().half()
        return w.contiguous(memory_format=torch.channels_last_3d) if self.cl else w.contiguous()

    @staticmethod
    def _point(x, w, b, z=None, relu=True):
        """A 1x1x1 convolution of a 5-D volume through its 4-D view: + bias, + z, ReLU, in one cuDNN call."""
        B, C, D, H, W = x.shape
        x4 = x.reshape(B, C, D * H, W)
        if z is not None:
            y = torch.cudnn_convolution_add_relu(x4, w, z.reshape(B, -1, D * H, W), 1.0, b, (1, 1), (0, 0), (1, 1), 1)
        elif relu:
            y = torch.cudnn_convolution_relu(x4, w, b, (1, 1), (0, 0), (1, 1), 1)
        else:
            y = F.conv2d(x4, w, b)
        return y.reshape(B, -1, D, H, W)

    @staticmethod
    def _depth_max(x):
        """The max over depth of a channels-last volume by halving it: each step is one elementwise pass over two contiguous
        halves.  Exact (a max has no rounding).  torch.max over dim 2 steps a megabyte through memory per element in this layout
        and took 18 ms a batch on the first stage."""
        while x.shape[2] > 1:
            d = x.shape[2]; h = d // 2
            m = torch.maximum(x[:, :, :h], x[:, :, h:2 * h])
            if d % 2: m[:, :, 0] = torch.maximum(m[:, :, 0], x[:, :, d - 1])
            x = m
        return x[:, :, 0]

    def _three(self, x, w, b, stride, pad):
        if self.three:
            return torch.cudnn_convolution_relu(x, w, b, stride, pad, (1, 1, 1), 1)
        y = F.conv3d(x, w, None, stride, pad)
        return y.add_(b.reshape(1, -1, 1, 1, 1)).relu_()

    def forward(self, x):
        x = F.interpolate(self.norm(x[:, None]), size=self.size, mode='trilinear', align_corners=False).half()
        if self.cl: x = x.contiguous(memory_format=torch.channels_last_3d)
        # the stem has one input channel, so its tensors read as the default layout whatever is asked: its pooled output is put
        # in channels-last here, once (64 channels), or every later call would convert its own input or residual
        x = self.pool(self._three(x, self.stem_w, self.stem_b, self.stem_stride, self.stem_pad))
        if self.cl: x = x.contiguous(memory_format=torch.channels_last_3d)
        pooled = []
        for k, stride, down, dstride, last in self.blocks:
            g = lambda n: getattr(self, k + n)                                        # noqa: E731
            out = self._point(x, g('w1'), g('b1'))
            out = self._three(out, g('w2'), g('b2'), (stride,) * 3, (1, 1, 1))
            if down:
                xs = x
                if dstride != 1:                                                      # one small copy, kept in the layout the view needs
                    xs = x[:, :, ::dstride, ::dstride, ::dstride].contiguous(memory_format=torch.channels_last_3d if self.cl else torch.contiguous_format)
                res = self._point(xs, g('wd'), g('bd'), relu=False)
            else:
                res = x
            x = self._point(out, g('w3'), g('b3'), z=res)
            if last: pooled.append(self._depth_max(x) if self.cl else torch.max(x, dim=2)[0])
        return self.decoder(pooled)


class CuFused2(CuFused):
    """CuFused with two more sources of traffic removed (found in its profile, 30 ms a tile):
      * the stem ran in the default layout (one input channel reads as either layout), so cuDNN converted its input and its
        1.6 GB output on every call, and the pooled result was converted again.  Here the input carries a second, zero channel
        with zero weights (the same sum), which makes the layout channels-last for real, and the (1, 3, 3) max-pool runs as a 2-D
        pool on the (B*D, C, H, W) view, which PyTorch pools in channels-last without converting;
      * the shortcut convolution's bias was added in a pass of its own: it is added to the bias of the block's last convolution
        instead, which takes the shortcut as cuDNN's z (relu(conv + (b3 + bd) + shortcut), the same sum).
    """

    def __init__(self, model):
        super().__init__(model, cl=True, three=True)
        w = self.stem_w
        self.register_buffer('stem_w2', torch.cat([w, torch.zeros_like(w)], 1).contiguous(memory_format=torch.channels_last_3d))
        for k, _, down, _, _ in self.blocks:
            if down: setattr(self, k + 'b3', (getattr(self, k + 'b3').float() + getattr(self, k + 'bd').float()).half())

    def forward(self, x):
        x = F.interpolate(self.norm(x[:, None]), size=self.size, mode='trilinear', align_corners=False).half()
        x = F.pad(x.permute(0, 2, 3, 4, 1), (0, 1)).permute(0, 4, 1, 2, 3)              # (B, 2, D, H, W) in channels-last memory
        x = torch.cudnn_convolution_relu(x, self.stem_w2, self.stem_b, self.stem_stride, self.stem_pad, (1, 1, 1), 1)
        B, C, D, H, W = x.shape
        x = F.max_pool2d(x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W), 3, 2, 1)
        x = x.reshape(B, D, C, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)
        pooled = []
        for k, stride, down, dstride, last in self.blocks:
            g = lambda n: getattr(self, k + n)                                        # noqa: E731
            out = self._point(x, g('w1'), g('b1'))
            out = self._three(out, g('w2'), g('b2'), (stride,) * 3, (1, 1, 1))
            if down:
                xs = x
                if dstride != 1:
                    xs = x[:, :, ::dstride, ::dstride, ::dstride].contiguous(memory_format=torch.channels_last_3d)
                b, c, d, h, w = xs.shape
                res = F.conv2d(xs.reshape(b, c, d * h, w), g('wd')).reshape(b, -1, d, h, w)
            else:
                res = x
            x = self._point(out, g('w3'), g('b3'), z=res)
            if last: pooled.append(self._depth_max(x))
        return self.decoder(pooled)


def build(name, model, device='cuda'):
    """-> (callable, module or None).  name as in the module docstring."""
    base, _, comp = name.partition('+')
    amp = True
    if base == 'stock':
        mod = model
    elif base == 'cl3':
        mod = cl3(copy.deepcopy(model))
    elif base == 'fuse':
        mod = fused(model).to(device)
    elif base == 'fuse_cl3':
        mod = cl3(fused(model).to(device))
    elif base in ('half', 'half_cl3'):
        f = fused(model).to(device)
        f.backbone.half(); f.decoder.half()                       # normalization stays float32
        if base == 'half_cl3':
            cl3(f)
        mod, amp = Split(f, base == 'half_cl3').eval(), False
    elif base in ('cufuse', 'cufuse_nchw', 'cufuse_p'):                # _nchw: NCDHW layout; _p: only the 1x1x1 convolutions fused
        mod, amp = CuFused(model, cl=base != 'cufuse_nchw', three=base != 'cufuse_p').to(device).eval(), False
    elif base == 'cufuse2':
        mod, amp = CuFused2(model).to(device).eval(), False
    elif base == 'fold2d':
        import ahn_fold
        mod = ahn_fold.fold(copy.deepcopy(model).cpu().eval(), 'stack').to(device).to(memory_format=torch.channels_last)
    else:
        raise ValueError(name)
    run = mod
    if comp == 'c':
        run = torch.compile(mod, dynamic=False)
    elif comp == 'ra':
        run = torch.compile(mod, dynamic=False, mode='reduce-overhead')
    elif comp:
        raise ValueError(name)

    def call(x):
        if amp:
            with torch.autocast('cuda', dtype=torch.float16):
                return run(x)
        return run(x)
    return call, mod
