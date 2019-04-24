import os

import torch
from torchtext.data import Field

from onmt.inputters.datareader_base import DataReaderBase

try:
    import numpy as np
    import cv2
    import torchvision
except ImportError:
    np = None
    cv2 = None
    torchvision = None


class VidDataReader(DataReaderBase):
    """Read feature vector data from disk.

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing ``np`` fails.
    """

    def __init__(self):
        self._check_deps()

    @classmethod
    def from_opt(cls, opt):
        return cls()

    @classmethod
    def _check_deps(cls):
        if np is None:
            cls._raise_missing_dep("np", "cv2")

    def read(self, vids, side, vid_dir=None):
        """Read data into dicts.

        Args:
            vids (str or Iterable[str]): Sequence of video paths or
                path to file containing feature video paths.
                In either case, the filenames may be relative to ``vid_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            vid_dir (str): Location of source vectors. See ``vids``.

        Yields:
            A dictionary containing feature vector data.
        """

        if isinstance(vids, str):
            vids = DataReaderBase._read_file(vids)

        for i, filename in enumerate(vids):
            filename = filename.decode("utf-8").strip()
            vid_path = os.path.join(vid_dir, filename)
            if not os.path.exists(vid_path):
                vid_path = filename

            assert os.path.exists(vid_path), \
                'vid path %s not found' % filename

            # vid = torch.load(vid_path)
            yield {side: vid_path, side + "_path": filename, "indices": i}


def vid_sort_key(ex):
    """Sort using the length of the vector sequence."""
    return torch.load(ex.src[0])[3].shape[1]


class VidSeqField(Field):
    """Defines an vector datatype and instructions for converting to Tensor.

    See :class:`Fields` for attribute descriptions.
    """

    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=False, batch_first=False, pad_index=0,
                 is_target=False):
        super(VidSeqField, self).__init__(
            sequential=True, use_vocab=False, init_token=None,
            eos_token=None, fix_length=False, dtype=torch.float,
            preprocessing=preprocessing, postprocessing=postprocessing,
            lower=False, tokenize=None, include_lengths=include_lengths,
            batch_first=batch_first, pad_token=pad_index, unk_token=None,
            pad_first=False, truncate_first=False, stop_words=None,
            is_target=is_target
        )
        self.xform = torchvision.transforms.ToTensor()
        self.means_2 = torch.load("/data/sd0/here/yt2t_2//means_2.pth")
        self.stds_2 = torch.load("/data/sd0/here/yt2t_2/stds_2.pth")

    def pad(self, minibatch):
        """Pad a batch of examples to the length of the longest example.

        Args:
            minibatch (List[torch.FloatTensor]): A list of audio data,
                each having shape ``(len, n_feats, feat_dim)``
                where len is variable.

        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape
                ``(batch_size, max_len, n_feats, feat_dim)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """

        assert not self.pad_first and not self.truncate_first \
            and not self.fix_length and self.sequential
        minibatch = list(minibatch)
        feats = [torch.load(p[0]) for p in minibatch]
        nL = len(feats[0])
        # process the non-vector features (i.e. the feature maps)
        lens_per_layer = [
            [feat[i].shape[1] for feat in feats]
            for i in range(nL - 1)
        ]
        max_l_per_layer = [max(ls) for ls in lens_per_layer]
        padded_feats = []
        bsz = len(minibatch)
        for layer_l in range(nL - 1):
            feats_l = [feat[layer_l] for feat in feats]
            nc, _, nx, ny = feats_l[0].shape
            padded = torch.zeros(
                (bsz, nc, max_l_per_layer[layer_l], nx, ny),
                dtype=torch.float32
            )
            for i, (feat, len_) in enumerate(
                    zip(feats_l, lens_per_layer[layer_l])):
                padded[i, :, :len_, :, :] = feat
            padded_feats.append(padded)

        feats_f = [feat[nL-1] for feat in feats]
        padded_feats.append(torch.stack(feats_f, 0))

        padded_feats[2] = (padded_feats[2] - self.means_2) / self.stds_2

        if self.include_lengths:
            return (padded_feats, lens_per_layer)
        return padded_feats

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.

        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): See `Field.numericalize`.
        """

        assert self.use_vocab is False
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)\
                .transpose(0, 1)
        # arr = arr.to(device)
        arr = [a.to(device) for a in arr]

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if self.sequential and self.batch_first:  # this is wrong
            arr = [a.transpose(0, 1) for a in arr]
        if self.sequential:
            arr = [a.contiguous() for a in arr]

        if self.include_lengths:
            return arr, lengths
        return arr


def vid_fields(**kwargs):
    vec = VidSeqField(pad_index=0, include_lengths=True)
    return vec
