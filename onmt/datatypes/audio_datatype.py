# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch
from torchtext.data import Field
try:
    import torchaudio
    import librosa
    import numpy as np
except ImportError:
    torchaudio, librosa, np = None, None, None

from onmt.datatypes.reader_base import DatasetReaderBase, \
    missing_dependency_exception
from onmt.datatypes.datatype_base import Datatype


def sort_key(ex):
    """Sort using duration time of the sound spectrogram."""
    return ex.src.size(1)


class AudioDatasetReader(DatasetReaderBase):
    """Create a dataset reader.

    Args:
        sample_rate (int): sample_rate.
        window_size (float) : window size for spectrogram in seconds.
        window_stride (float): window stride for spectrogram in seconds.
        window (str): window type for spectrogram generation.
        normalize_audio (bool): subtract spectrogram by mean and divide
            by std or not.
        truncate (int or NoneType): maximum audio length
            (0 or None for unlimited).

    """
    def __init__(self, sample_rate, window_size,
                 window_stride, window, normalize_audio, truncate=None):
        super(AudioDatasetReader, self).__init__()
        if not self._deps_imported():
            missing_dependency_exception(AudioDatasetReader.__name__,
                                         'torchaudio', 'librosa', 'numpy')
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize_audio = normalize_audio
        self.truncate = truncate

    @classmethod
    def from_opt(cls, opt):
        return cls(opt.sample_rate, opt.window_size,
                   opt.window_stride, opt.window, True,
                   None)

    @staticmethod
    def _deps_imported():
        return not any([torchaudio is None, librosa is None, np is None])

    def extract_features(self, audio_path):
        # torchaudio loading options recently changed. It's probably
        # straightforward to rewrite the audio handling to make use of
        # up-to-date torchaudio, but in the meantime there is a legacy
        # method which uses the old defaults
        sound, sample_rate_ = torchaudio.legacy.load(audio_path)
        if self.truncate and self.truncate > 0:
            if sound.size(0) > self.truncate:
                sound = sound[:self.truncate]

        assert sample_rate_ == self.sample_rate, \
            'Sample rate of %s != -sample_rate (%d vs %d)' \
            % (audio_path, sample_rate_, self.sample_rate)

        sound = sound.numpy()
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # average multiple channels

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, _ = librosa.magphase(d)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize_audio:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    def read(self, audio_files, side, src_dir=None):
        """Read sound files.

        Args:
            audio_files (str, List[str]): Either a file of one file path per
                line (either existing path or path relative to `src_dir`)
                or a list thereof.
            side (str): 'src' or 'tgt'.
            src_dir (str): location of source audio files.

        Yields:
            a dictionary containing audio data for each line.

        """
        if isinstance(audio_files, str):
            audio_files = self._read_file(audio_files)

        for i, filename in enumerate(tqdm(audio_files)):
            filename = filename.strip()
            audio_path = os.path.join(src_dir, filename)
            if not os.path.exists(audio_path):
                audio_path = filename

            assert os.path.isfile(audio_path), \
                'audio path {:s} not found'.format(filename)

            spect = self.extract_features(audio_path)

            yield {side: spect, side + '_path': filename,
                   side + '_lengths': spect.size(1), 'indices': i}


def make_audio(data, vocab):
    """ batch audio data """
    nfft = data[0].size(0)
    t = max([t.size(1) for t in data])
    sounds = torch.zeros(len(data), 1, nfft, t)
    for i, spect in enumerate(data):
        sounds[i, :, :, 0:spect.size(1)] = spect
    return sounds


def fields(base_name, **kwargs):
    audio = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=make_audio, sequential=False)

    length = Field(use_vocab=False, dtype=torch.long, sequential=False)

    return [(base_name, audio), (base_name + "_lengths", length)]


audio_datatype = Datatype("audio", AudioDatasetReader, sort_key, fields)
