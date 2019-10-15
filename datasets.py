import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def video_len(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoFeaturesDataset(data.Dataset):
    def __init__(self, features_folder, list_file, num_frames, sampling_strategy, min_dataset_size=1024):
        assert sampling_strategy in ['TSNTrain', 'TSNVal', 'ALL']

        self.features_folder = features_folder
        self.list_file = list_file
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy

        self.feature_template = 'img_{:05d}.t7'

        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        self.num_classes = len(set([record.label for record in self.video_list]))

        # Repeat dataset if less than min_size
        self.dataset_size = min(len(self.video_list), min_dataset_size)

    def tsn_sample(self, record, is_val=False):
        segment_duration = record.video_len // self.num_frames

        if is_val:
            frame_idx = np.multiply(list(range(self.num_frames)), segment_duration) + 0.5 * segment_duration
        else:
            frame_idx = np.multiply(list(range(self.num_frames)), segment_duration)
            frame_idx += randint(segment_duration, size=self.num_frames)

        return frame_idx

    def __getitem__(self, idx):
        idx %= len(self.video_list)
        record = self.video_list[idx]

        if self.sampling_strategy == 'TSNTrain':
            frame_idx = self.tsn_sample(record, is_val=False)
        elif self.sampling_strategy == 'TSNVal':
            frame_idx = self.tsn_sample(record, is_val=True)
        elif self.sampling_strategy == 'ALL':
            frame_idx = np.arange(record.video_len)

        return self.read(record, frame_idx)

    def read(self, record, frame_idx):
        frames = list()
        frame_idx = frame_idx.astype(int)

        for idx in frame_idx:
            feat_path = os.path.join(self.features_folder, record.path, self.feature_template.format(idx))
            feat = torch.load(feat_path)
            frames.append(feat)

        frames = torch.stack(frames)

        return {'frames': frames, 'labels': record.label, 'idx': frame_idx}

    def __len__(self):
        return len(self.dataset_size)
