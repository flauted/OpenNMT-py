# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch
from torchtext.data import Field

from onmt.inputters.dataset_base import DatasetBase

# imports of datatype-specific dependencies
try:
    import torchaudio
    import librosa
    import numpy as np
except ImportError:
    torchaudio, librosa, np = None, None, None


class AudioDataset(DatasetBase):
    @staticmethod
    def _check_deps():
        if any([torchaudio is None, librosa is None, np is None]):
            AudioDataset._raise_missing_dep(
                "torchaudio", "librosa", "numpy")

    @staticmethod
    def sort_key(ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def extract_features(audio_path, sample_rate, truncate, window_size,
                         window_stride, window, normalize_audio):
        # torchaudio loading options recently changed. It's probably
        # straightforward to rewrite the audio handling to make use of
        # up-to-date torchaudio, but in the meantime there is a legacy
        # method which uses the old defaults
        AudioDataset._check_deps()
        sound, sample_rate_ = torchaudio.legacy.load(audio_path)
        if truncate and truncate > 0:
            if sound.size(0) > truncate:
                sound = sound[:truncate]

        assert sample_rate_ == sample_rate, \
            'Sample rate of %s != -sample_rate (%d vs %d)' \
            % (audio_path, sample_rate_, sample_rate)

        sound = sound.numpy()
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # average multiple channels

        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)
        # STFT
        d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
        spect, _ = librosa.magphase(d)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if normalize_audio:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    @classmethod
    def make_examples(
        cls,
        data,
        src_dir,
        side,
        sample_rate,
        window_size,
        window_stride,
        window,
        normalize_audio,
        truncate=None
    ):
        """
        Args:
            data: sequence of audio paths or path containing these sequences
            src_dir (str): location of source audio files.
            side (str): 'src' or 'tgt'.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Yields:
            a dictionary containing audio data for each line.
        """
        assert src_dir is not None and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        if isinstance(data, str):
            data = cls._read_file(data)

        for i, line in enumerate(tqdm(data)):
            audio_path = os.path.join(src_dir, line.strip())
            if not os.path.exists(audio_path):
                audio_path = line.strip()

            assert os.path.exists(audio_path), \
                'audio path %s not found' % (line.strip())

            spect = AudioDataset.extract_features(
                audio_path, sample_rate, truncate, window_size,
                window_stride, window, normalize_audio
            )

            yield {side: spect, side + '_path': line.strip(),
                   side + '_lengths': spect.size(1), 'indices': i}


def batch_audio(data, vocab):
    """ batch audio data """
    nfft = data[0].size(0)
    t = max([t.size(1) for t in data])
    sounds = torch.zeros(len(data), 1, nfft, t)
    for i, spect in enumerate(data):
        sounds[i, :, :, 0:spect.size(1)] = spect
    return sounds


def audio_fields(base_name, **kwargs):
    audio = Field(
        use_vocab=False, dtype=torch.float,
        postprocessing=batch_audio, sequential=False)

    length = Field(use_vocab=False, dtype=torch.long, sequential=False)

    return [(base_name + "_lengths", length)], [(base_name, audio)]
