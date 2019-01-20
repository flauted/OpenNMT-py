# -*- coding: utf-8 -*-

import os

import torch
from torchtext.data import Field

try:
    from PIL import Image
    from torchvision import transforms
    import cv2
except ImportError:
    Image, transforms, cv2 = None, None, None

from onmt.datatypes.reader_base import DatasetReaderBase, \
    missing_dependency_exception
from onmt.datatypes.datatype_base import Datatype


CHANNEL_DIM, ROW_DIM, COL_DIM = 0, 1, 2


def sort_key(ex):
    """Sort using the size of the image: (width, height)."""
    return ex.src.size(COL_DIM), ex.src.size(ROW_DIM)


class ImageDatasetReader(DatasetReaderBase):
    """Create an image dataset reader.

    Args:
        truncate: maximum img size given as (rows, cols)
            ((0,0) or None for unlimited)
        channel_size (int): Number of image channels. Set to 1 for
            grayscale.

    """
    def __init__(self, truncate=None, channel_size=3):
        super(ImageDatasetReader, self).__init__()
        if not self._deps_imported():
            missing_dependency_exception(ImageDatasetReader.__name__,
                                         'PIL', 'torchvision', 'cv2')
        self.truncate = truncate
        self.channel_size = channel_size

    @classmethod
    def from_opt(cls, opt):
        return cls(channel_size=opt.image_channel_size)

    @staticmethod
    def _deps_imported():
        return Image is not None and transforms is not None and cv2 is not None

    def read(self, im_files, side, src_dir=None):
        """Read images.

        Args:
            im_files (str, List[str]): Either a file of one file path per
                line (either existing path or path relative to `src_dir`)
                or a list thereof.
            src_dir (str): location of source images
            side (str): 'src' or 'tgt'

        Yields:
            a dictionary containing image data, path and index for each line.

        """
        if isinstance(im_files, str):
            im_files = self._read_file(im_files)

        for i, filename in enumerate(im_files):
            filename = filename.strip()
            img_path = os.path.join(src_dir, filename)
            if not os.path.isfile(img_path):
                img_path = filename

            assert os.path.isfile(img_path), \
                'img path {:s} not found'.format(filename)

            if self.channel_size == 1:
                img = transforms.ToTensor()(
                    Image.fromarray(cv2.imread(img_path, 0)))
            else:
                img = transforms.ToTensor()(Image.open(img_path))
            if self.truncate and self.truncate != (0, 0):
                if not (img.size(ROW_DIM) <= self.truncate[0]
                        and img.size(COL_DIM) <= self.truncate[1]):
                    continue
            yield {side: img, side + '_path': filename, 'indices': i}


def make_img(data, vocab):
    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.zeros(len(data), c, h, w).fill_(1)
    for i, img in enumerate(data):
        imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
    return imgs


def fields(base_name, **kwargs):
    img = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=make_img, sequential=False)
    return [(base_name, img)]


image_datatype = Datatype("img", ImageDatasetReader, sort_key, fields)
