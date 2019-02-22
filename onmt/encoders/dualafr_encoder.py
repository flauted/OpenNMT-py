import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase

try:
    from src.i3dpt import I3D
except ImportError:
    I3D = None


if I3D:
    class I3DFE(I3D, nn.Module):
        MIN_VID_LEN = 10

        def __init__(self, *args, **kwargs):
            if I3D is None:
                raise Exception("Could not load I3D. Did you install it?")
            super(I3DFE, self).__init__(*args, **kwargs)

        def forward(self, inp):
            # Preprocessing
            returns = []
            if inp.shape[2] < self.MIN_VID_LEN:
                inp = torch.cat(
                    [inp, torch.zeros(
                        (inp.shape[0], inp.shape[1],
                         self.MIN_VID_LEN - inp.shape[2],
                         inp.shape[3], inp.shape[4]),
                        dtype=inp.dtype, device=inp.device)], 2)
            out = self.conv3d_1a_7x7(inp)
            out = self.maxPool3d_2a_3x3(out)
            returns.append(out)
            out = self.conv3d_2b_1x1(out)
            out = self.conv3d_2c_3x3(out)
            out = self.maxPool3d_3a_3x3(out)
            returns.append(out)
            out = self.mixed_3b(out)
            out = self.mixed_3c(out)
            out = self.maxPool3d_4a_3x3(out)
            returns.append(out)
            out = self.mixed_4b(out)
            out = self.mixed_4c(out)
            out = self.mixed_4d(out)
            out = self.mixed_4e(out)
            out = self.mixed_4f(out)
            out = self.maxPool3d_5a_2x2(out)
            returns.append(out)
            out = self.mixed_5b(out)
            out = self.mixed_5c(out)
            out = self.avg_pool(out)
            out = self.dropout(out)
            # putting the mean before conv3d gives the same output on the
            # I3D demo, so it would seem this is commutative with the conv3d
            out = out.mean(-1, keepdim=True)
            # out = self.conv3d_0c_1x1(out)
            returns.append(out)
            return returns


class DualAFREncoder(EncoderBase):
    def __init__(self, embeddings):
        super(DualAFREncoder, self).__init__()
        self.embeddings = embeddings
        # TODO: Expose flow
        self.model = I3DFE(num_classes=400, modality="rgb")
        # TODO: Expose this path!
        self.model.load_state_dict(
            torch.load("/home/dylan/code/kinetics_i3d_pytorch/model/model_rgb.pth")
        )

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        return cls(embeddings)

    def forward(self, src, lengths=None):
        with torch.no_grad():
            feats = self.model(src)
            return feats[-1].mean(2).squeeze(-1).squeeze(-1).unsqueeze(0), \
                   feats[:-1], lengths
