import argparse
import os
from math import floor

import tqdm
from multiprocessing import Manager
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import pretrainedmodels
from pretrainedmodels.utils import TransformImage
try:
    from src.i3dpt import I3D
except ImportError:
    I3D = None


FINISHED = "finished"  # end-of-queue


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.zeropad1 = nn.ConstantPad3d((0, 0, 0, 0, 0, 1), 0)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.zeropad5 = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.relu = nn.ReLU()
        self.cube = torch.from_numpy(
            np.load("/home/dylan/Downloads/c3d_mean.npy"))\
            .to(torch.float32)

    def forward(self, x):
        x = x.float()
        x -= self.cube
        x = x[:, :, :, 8:120, 30:142]
        results = []
        h = self.conv1(x)
        h = self.relu(self.pool1(h))

        h = self.conv2(h)
        h = self.relu(self.pool2(self.zeropad1(h)))
        results.append(h)

        h = self.relu(self.conv3a(h))
        h = self.conv3b(h)
        h = self.relu(self.pool3(self.zeropad1(h)))
        results.append(h)

        h = self.relu(self.conv4a(h))
        h = self.conv4b(h)
        h = self.relu(self.pool4(self.zeropad1(h)))
        results.append(h)

        h = self.relu(self.conv5a(h))
        h = self.conv5b(h)
        h = self.relu(self.pool5(self.zeropad5(h)))
        results.append(h)

        # h = h.flatten(2).mean(2)
        # results.append(h)

        bsz = h.shape[0]
        h = h.view(bsz, -1, 8192)
        h = self.relu(self.fc6(h))
        results.append(self.relu(self.fc7(h)))
        return results

    # def tops(self, x5):
    #     bsz = x5.shape[0]
    #     h = self.relu(self.fc6(x5.view(bsz, -1, 8192)))
    #     return self.relu(self.fc7(h))


class C3DFeatureExtractor(nn.Module):
    def __init__(self):
        super(C3DFeatureExtractor, self).__init__()
        self.model = C3D()
        sd = torch.load("/home/dylan/Downloads/c3d.pickle")
        del sd["fc8.weight"]
        del sd["fc8.bias"]
        self.model.load_state_dict(sd)

    def to(self, *args, **kwargs):
        self.model.cube = self.model.cube.to(*args, **kwargs)
        return super(C3DFeatureExtractor, self).to(*args, **kwargs)

    def forward(self, x):
        # x_2fps = x[:, :, ::10]
        # if x.shape[2] % 10 != 1:
        #     x_2fps = torch.cat((x_2fps, x[:, :, -1].unsqueeze(2)), 2)
        # outps = self.model(x_2fps)
        # feat_maps = outps[:4]
        fvs = [[] for _ in range(5)]

        lt16 = x.shape[2] < 16
        while x.shape[2] < 16:
            x = torch.cat([x, x], 2)

        if lt16:
            x = x[:, :, :16]

        nframes = x.shape[2]
        ct = 0
        while True:
            start = 8 * ct
            if start >= nframes:
                break
            end = 8 * (ct + 2)
            if end > nframes:
                end = nframes
                start = end - 16
                if start < 0:
                    start = 0
                # bsz, nf, nz, nx, ny = x.shape
                # x = torch.cat(
                #     [x, torch.zeros((bsz, nf, end - nz, nx, ny),
                #                     dtype=x.dtype, device=x.device)],
                #     2)
            seg = x[:, :, start:end]
            for i, fv in enumerate(self.model(seg)):
                fvs[i].append(fv)
            ct += 1
            if end == nframes:
                break
        feat_maps = [torch.stack(vs, -1).mean(-1) for vs in fvs]
        # fv = torch.stack(fvs, -1).mean(-1)
        # feat_maps.append(self.model.tops(fv).squeeze(1))
        # bsz = fv.shape[0]
        # feat_maps = [fm / fm.view(bsz, -1).norm(dim=1, keepdim=True)
        #              for fm in feat_maps]
        # set average mean to 1 based on yt2t means
        # layer_means = [35.7437, -15.8307, -2.7768, -0.2413, -0.2413]
        # layer_stds = [119.5741, 124.0031, 8.6424, 0.5234, 0.5234]
        # feat_maps[:3] = \
        #     [(fm - mean_l) / (4 * std_l)
        #      for fm, mean_l, std_l
        #      in zip(feat_maps[:3], layer_means, layer_stds)]
        # feat_maps[4] = (feat_maps[4] - layer_means[4]) / layer_stds[4]
        return feat_maps


def read_to_imgs(file):
    """Yield images and their frame number from a video file."""
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    idx = 0
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image, idx
        idx += 1
        success, image = vidcap.read()


# TODO: Expose nframes
def read_to_vids(file, cols, rows, nframes=16):
    """Yields a video."""
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    frames = []
    ct = 0
    fps_cap = vidcap.get(cv2.CAP_PROP_FPS)
    f_cap = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_desired = 10
    s = f_cap / fps_cap  # float
    f_desired = int(floor(fps_desired * s))  # float
    every_nth_float = f_cap / f_desired
    include_frames = [floor(every_nth_float * i) for i in range(f_desired)] \
                      + [f_cap - 1]
    while success:
        if ct in include_frames:
            image = cv2.resize(image, (cols, rows))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        success, image = vidcap.read()
        ct += 1
    return np.stack(frames, 0), None


def vid_len(path):
    """Return the length of a video."""
    return int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))


class VidDset(object):
    """For each video, yield its frames."""
    def __init__(self, model, root_dir, filenames, model_type="cnn"):
        self.root_dir = root_dir
        self.filenames = filenames
        self.paths = [os.path.join(self.root_dir, f) for f in self.filenames]
        self.return_vids = model_type == "i3d" or \
                           model_type == "c3d" or \
                           model_type == "none"
        if model_type == "cnn":
            self.xform = TransformImage(model)
        else:
            self.xform = lambda x: torch.from_numpy(x).permute(3, 0, 1, 2)

        self.current = 0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        path = self.paths[i]
        if self.return_vids:
            return [(path, _, self.xform(vid))
                    for vid, _ in [read_to_vids(path, 171, 128)]]
        else:
            return ((path, idx, self.xform(Image.fromarray(img)))
                    for img, idx in read_to_imgs(path))

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


def collate_tensor(batch, return_vids):
    if return_vids:
        # pad the videos
        vids = batch[-1]
        max_len = max(v.shape[1] for v in vids)
        bsz = len(batch[-1])
        c, _, x, y = batch[-1][0].shape
        vid = torch.zeros((bsz, c, max_len, x, y), dtype=torch.uint8)
        for i, v in enumerate(vids):
            vid[i, :, :v.shape[1], :, :] = v
        batch[-1] = vid
    else:
        batch[-1] = torch.stack(batch[-1], 0)


def batch(dset, batch_size):
    """Collate frames into batches of equal length."""
    batch = [[], [], []]
    batch_ct = 0
    for seq in dset:
        for path, idx, img in seq:
            if batch_ct == batch_size:
                collate_tensor(batch, dset.return_vids)
                yield batch
                batch = [[], [], []]
                batch_ct = 0
            batch[0].append(path)
            batch[1].append(idx)
            batch[2].append(img)
            batch_ct += 1
    if batch_ct != 0:
        collate_tensor(batch, dset.return_vids)
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


class VidSaver(object):
    def __init__(self, out_path, finished_queue):
        self.out_path = out_path
        self.finished_queue = finished_queue

    def save(self, path, feats):
        torch.save(feats, path)

    def push(self, paths, idxs, feats):
        for path_b, vid_b in zip(paths, feats):
            save_path, vid_id = Reconstructor.name_(path_b, self.out_path)
            self.save(save_path, vid_b)
            self.finished_queue.put((vid_id, vid_len(path_b)))

    def flush(self):
        pass


class I3DSaver(object):
    def __init__(self, out_path, finished_queue):
        self.out_path = out_path
        self.finished_queue = finished_queue

    def save(self, path, feats):
        torch.save(feats, path)

    def push(self, paths, idxs, feats):
        for b, path_b in enumerate(paths):
            save_path, vid_id = Reconstructor.name_(path_b, self.out_path)
            feats_b = [f[b] for f in feats]
            self.save(save_path, feats_b)
            self.finished_queue.put((vid_id, vid_len(path_b)))

    def flush(self):
        pass


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
                if i - 1 < 0:
                    name = self.path
                else:
                    name = paths[i-1]
                save_path, vid_id = self.name(name)
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


class VidFeatureExtractor(object):
    model = None


def run(device_id, world_size, root_dir, batch_size_per_device,
        feats_queue, files, model_type="cnn"):
    """Process a disjoint subset of the videos on each device."""
    if world_size > 1:
        these_files = [f for i, f in enumerate(files)
                       if i % world_size == device_id]
    else:
        these_files = files
    if model_type == "cnn":
        fe = FeatureExtractor()
    elif model_type == "i3d":
        fe = I3DFeatureExtractor()
    elif model_type == "c3d":
        fe = C3DFeatureExtractor()
    else:
        fe = VidFeatureExtractor()
    dset = VidDset(fe.model, root_dir, these_files, model_type=model_type)
    if model_type == "none":
        for samp in batch(dset, batch_size_per_device):
            paths, idxs, images = samp
            feats_queue.put((paths, idxs, images))
        feats_queue.put(FINISHED)
        return
    dev = torch.device("cuda", device_id) \
        if device_id >= 0 else torch.device("cpu")
    fe.to(dev)
    fe = fe.eval()
    with torch.no_grad():
        for samp in batch(dset, batch_size_per_device):
            paths, idxs, images = samp
            images = images.to(dev)
            feats = fe(images)
            if torch.is_tensor(feats):
                feats = feats.to("cpu")
            else:
                feats = [f.to("cpu") for f in feats]
            feats_queue.put((paths, idxs, feats))
    feats_queue.put(FINISHED)
    return


def saver(out_path, model_type, feats_queue, finished_queue):
    if model_type == "cnn":
        rc = Reconstructor(out_path, finished_queue)
    elif model_type == "i3d" or model_type == "c3d":
        rc = I3DSaver(out_path, finished_queue)
    else:
        rc = VidSaver(out_path, finished_queue)
    while True:
        item = feats_queue.get()
        if item == FINISHED:
            rc.flush()
            finished_queue.put(FINISHED)
            return
        else:
            paths, idxs, feats = item
            rc.push(paths, idxs, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Directory of videos.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory for output features.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of devices to run on.")
    parser.add_argument("--batch_size_per_device", type=int, default=512)
    parser.add_argument("--model_type", type=str,
                        choices=["cnn", "i3d", "c3d", "none"])
    opt = parser.parse_args()

    batch_size_per_device = opt.batch_size_per_device
    root_dir = opt.root_dir
    out_path = opt.out_dir
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

    files = list(sorted(list(os.listdir(root_dir))))
    files = [f for f in files
             if os.path.basename(Reconstructor.name_(f, out_path)[0])
             not in os.listdir(out_path)]

    procs.append(mp.Process(
        target=finished_watcher,
        args=(finished_queue, world_size, root_dir, files),
        daemon=False
    ))
    procs[0].start()

    if world_size >= 1:
        feat_queues = [manager.Queue(2) for _ in range(world_size)]
        for feats_queue, device_id in zip(feat_queues, range(world_size)):
            # each device has its own saver so that reconstructing is easier
            # mgr = Manager()
            procs.append(mp.Process(
                target=run,
                args=(device_id, world_size, root_dir,
                      batch_size_per_device, feats_queue, files,
                      opt.model_type),
                daemon=True))
            procs[-1].start()
            procs.append(mp.Process(
                target=saver,
                args=(out_path, opt.model_type, feats_queue, finished_queue),
                daemon=True))
            procs[-1].start()
    else:
        feats_queue = manager.Queue()
        procs.append(mp.Process(
            target=run,
            args=(-1, 1, root_dir,
                  batch_size_per_device, feats_queue, files,
                  opt.model_type),
            daemon=True))
        procs[-1].start()
        procs.append(mp.Process(
            target=saver,
            args=(out_path, opt.model_type, feats_queue, finished_queue),
            daemon=True))
        procs[-1].start()

    for p in procs:
        p.join()
