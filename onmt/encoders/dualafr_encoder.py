import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask

try:
    from src.i3dpt import Unit3Dpy, MaxPool3dTFPadding, Mixed
except ImportError:
    Unit3Dpy, MaxPool3dTFPadding, Mixed = None, None, None


def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DualAFREncoder(EncoderBase):
    def __init__(self, embeddings):
        super(DualAFREncoder, self).__init__()
        # self.conv0 = nn.Sequential(
        #     nn.Conv3d(128, 512, (7, 7, 7), padding=(3, 3, 3)),
        #     nn.ReLU())
        # self.pad0 = nn.Sequential(
        #     nn.ConstantPad3d((0, 0, 0, 0, 0, 7), 0),
        #     nn.MaxPool3d(kernel_size=(8, 8, 8), stride=(8, 8, 8),
        #                  padding=(0, 4, 4)))
        #
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(256, 512, (5, 5, 5), padding=(2, 2, 2)),
        #     nn.ReLU())
        # self.pad1 = nn.Sequential(
        #     nn.ConstantPad3d((0, 0, 0, 0, 0, 3), 0),
        #     nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(4, 4, 4),
        #                  padding=(0, 2, 2)))
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(512, 512, (3, 3, 3), padding=(1, 1, 1)),
        #     nn.ReLU())
        # self.pad2 = nn.Sequential(
        #         nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0),
        #         nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.apply(init_weights)
        self.dp = nn.Dropout()
        # self.bn_pool5 = nn.BatchNorm3d(512, affine=False)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        return cls(embeddings)

    def forward(self, src, lengths=None):
        x0, x1, x2, x3 = src[:-1]
        # self.dp(x2)
        bsz, _, nf = src[-1].shape
        final = src[-1].view(1, bsz, nf)
        # final = final / final.norm(p=2, dim=-1, keepdim=True)

        # # layer 1
        # x0 = self.conv0(x0)
        #
        # _, nc, nz, nx, ny = src[0].shape
        # mask = sequence_mask(lengths[:, 0], max_len=nz)\
        #     .view(bsz, 1, nz, 1, 1)\
        #     .repeat(1, 1, 1, nx, ny)
        # x0.masked_fill_(~mask, 0)
        #
        # x0 = self.pad0(x0)
        #
        # # layer 2
        # x1 = self.conv1(x1)
        #
        # _, nc, nz, nx, ny = src[1].shape
        # mask = sequence_mask(lengths[:, 1], max_len=nz) \
        #     .view(bsz, 1, nz, 1, 1) \
        #     .repeat(1, 1, 1, nx, ny)
        # x1.masked_fill_(~mask, 0)
        #
        # x1 = self.pad1(x1)
        #
        # layer 3
        # x2 = self.conv2(x2)
        # x2 = self.dp(x2)
        #
        # _, nc, nz, nx, ny = src[2].shape
        # mask = sequence_mask(lengths[:, 2], max_len=nz) \
        #     .view(bsz, 1, nz, 1, 1) \
        #     .repeat(1, 1, 1, nx, ny)
        # x2.masked_fill_(~mask, 0)
        #
        # x2 = self.pad2(x2)

        # feats = [self.dp(x0),
        #          self.dp(x1),
        #          self.dp(x2),
        #          self.dp(x3)]
        feats = [x0, x1, x2, x3]
        return final, feats, lengths
