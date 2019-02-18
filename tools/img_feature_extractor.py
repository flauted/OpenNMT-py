import argparse
import os

import tqdm
from multiprocessing import Manager
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import pretrainedmodels
from pretrainedmodels.utils import TransformImage


FINISHED = "finished"  # end-of-queue


def read_to_imgs(file):
    """Yield images and their frame number from a video file."""
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    idx = 0
    while success:
        yield image, idx
        idx += 1
        success, image = vidcap.read()


def vid_len(path):
    """Return the length of a video."""
    return int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))


class VidDset(object):
    """For each video, yield its frames."""
    def __init__(self, model, root_dir, filenames):
        self.root_dir = root_dir
        self.filenames = filenames
        self.paths = [os.path.join(self.root_dir, f) for f in self.filenames]
        self.xform = TransformImage(model)

        self.current = 0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        path = self.paths[i]
        for img, idx in read_to_imgs(path):
            yield path, idx, self.xform(Image.fromarray(img))

    def __iter__(self):
        return self

    def next(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            self.current += 1
            return self[self.current - 1]

    def __next__(self):
        return self.next()


def batch(dset, batch_size):
    """Collate frames into batches of equal length."""
    batch = [[], [], []]
    batch_ct = 0
    for seq in dset:
        for path, idx, img in seq:
            if batch_ct == batch_size:
                batch[-1] = torch.stack(batch[-1], 0)
                yield batch
                batch = [[], [], []]
                batch_ct = 0
            batch[0].append(path)
            batch[1].append(idx)
            batch[2].append(img)
            batch_ct += 1
    if batch_ct != 0:
        batch[-1] = torch.stack(batch[-1], 0)
        yield batch


class FeatureExtractor(nn.Module):
    """Extract feature vectors from a batch of frames."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = pretrainedmodels.resnet152()
        self.FEAT_SIZE = 2048

    def forward(self, x):
        return self.model.avgpool(
            self.model.features(x)).view(-1, 1, self.FEAT_SIZE)


class Reconstructor(object):
    """Turn batches of feature vectors into sequences for each video.

    Assumes data is ordered (use one reconstructor per process).
    :func:`push()` batches in. When finished, :func:`flush()`
    the last sequence.
    """

    def __init__(self, out_path, finished_queue):
        self.out_path = out_path
        self.feats = None
        self.finished_queue = finished_queue

    def save(self, path, feats):
        np.save(path, feats.numpy())

    @staticmethod
    def name_(path, out_path):
        vid_path = path
        vid_fname = os.path.basename(vid_path)
        vid_id = os.path.splitext(vid_fname)[0]

        save_fname = vid_id + ".npy"
        save_path = os.path.join(out_path, save_fname)
        return save_path, vid_id

    def name(self, path):
        return self.name_(path, self.out_path)

    def push(self, paths, idxs, feats):
        start = 0
        for i, idx in enumerate(idxs):
            if idx == 0:
                if self.feats is None and i == 0:
                    # degenerate case
                    continue
                these_finished_seq_feats = feats[start:i]
                if self.feats is not None:
                    all_last_seq_feats = torch.cat(
                        [self.feats, these_finished_seq_feats], 0)
                else:
                    all_last_seq_feats = these_finished_seq_feats
                save_path, vid_id = self.name(paths[i-1])
                self.save(save_path, all_last_seq_feats)
                n_feats = all_last_seq_feats.shape[0]
                self.finished_queue.put((vid_id, n_feats))
                self.feats = None
                start = i
        # cache the features
        if self.feats is None:
            self.feats = feats[start:]
        else:
            self.feats = torch.cat([self.feats, feats[start:]], 0)
        self.path = paths[-1]

    def flush(self):
        if self.feats is not None:  # shouldn't be
            save_path, vid_id = self.name(self.path)
            self.save(save_path, self.feats)
            self.finished_queue.put((vid_id, self.feats.shape[0]))


def finished_watcher(finished_queue, world_size, root_dir, files):
    """Keep a progress bar of frames finished."""
    n_frames = sum(vid_len(os.path.join(root_dir, f)) for f in files)
    n_finished_frames = 0
    with tqdm.tqdm(total=n_frames, unit="Fr") as pbar:
        n_proc_finished = 0
        while True:
            item = finished_queue.get()
            if item == FINISHED:
                n_proc_finished += 1
                if n_proc_finished == world_size:
                    return
            else:
                vid_id, n_these_frames = item
                n_finished_frames += n_these_frames
                pbar.set_postfix(vid=vid_id)
                pbar.update(n_these_frames)


def run(device_id, world_size, root_dir, out_path, batch_size_per_device,
        finished_queue, files):
    """Process a disjoint subset of the videos on each device."""
    if world_size > 1:
        these_files = [f for i, f in enumerate(files)
                       if i % world_size == device_id]
    else:
        these_files = files
    fe = FeatureExtractor()
    dset = VidDset(fe.model, root_dir, these_files)
    rc = Reconstructor(out_path, finished_queue)
    dev = torch.device("cuda", device_id) \
        if device_id >= 0 else torch.device("cpu")
    fe.to(dev)
    fe = fe.eval()
    with torch.no_grad():
        for samp in batch(dset, batch_size_per_device):
            paths, idxs, images = samp
            images = images.to(dev)
            feats = fe(images).to("cpu")
            rc.push(paths, idxs, feats)
    rc.flush()
    finished_queue.put(FINISHED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Directory of videos.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory for output features.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of devices to run on.")
    parser.add_argument("--batch_size_per_device", type=int, default=512)
    opt = parser.parse_args()

    batch_size_per_device = opt.batch_size_per_device
    root_dir = opt.root_dir
    out_path = opt.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # mp queues don't work well between procs unless they're from a manager
    manager = Manager()
    finished_queue = manager.Queue()

    world_size = opt.world_size if torch.cuda.is_available() else -1

    mp = torch.multiprocessing.get_context("spawn")
    procs = []

    print("Starting processing. Progress bar startup can take some time, but "
          "processing will start in the meantime.")

    files = list(os.listdir(root_dir))
    files = [f for f in files
             if os.path.basename(Reconstructor.name_(f, out_path)[0])
             not in os.listdir(out_path)]

    procs.append(mp.Process(
        target=finished_watcher,
        args=(finished_queue, world_size, root_dir, files),
        daemon=True
    ))
    procs[0].start()

    if world_size > 1:
        for device_id in range(world_size):
            procs.append(mp.Process(
                target=run,
                args=(device_id, world_size, root_dir, out_path,
                      batch_size_per_device, finished_queue, files),
                daemon=True))
            procs[device_id + 1].start()
    elif world_size == 1:
        procs.append(mp.Process(
            target=run,
            args=(0, 1, root_dir, out_path,
                  batch_size_per_device, finished_queue, files),
            daemon=True))
        procs[1].start()
    else:
        procs.append(mp.Process(
            target=run,
            args=(-1, 1, root_dir, out_path,
                  batch_size_per_device, finished_queue, files),
            daemon=True))
        procs[1].start()

    for p in procs:
        p.join()
