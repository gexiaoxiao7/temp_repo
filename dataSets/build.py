import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.tld import TeacherDetection
import numpy as np
from PIL import Image
import cv2
import torch
import clip
from utils.tools import split_dataset
import torch.distributed as dist
import os
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,config,preprocess,device,ann_file,shot=0,type = 'train'):
        self.video_info = self.load_annotations(
            ann_file = ann_file,
            data_prefix = config.DATA.ROOT,
            num_frames = config.DATA.NUM_FRAMES,
            input_size = config.DATA.INPUT_SIZE,
            preprocess = preprocess,
            device = device,
            shot = shot,
            type = type,
            if_teacher = config.MODEL.IF_TEACHER,
            detector = TeacherDetection(config.MODEL.YOLO)
        )

    def prepare_frames(self, path, num_frames, if_teacher, detector, preprocess):
        if not os.path.exists(path):
            print(f"File {path} not found.")
            return None
        video_capture = cv2.VideoCapture(path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_ids = np.linspace(0, total_frames - 2, num_frames)
        frame_ids = np.floor(frame_ids).astype(int)
        for i in range(total_frames+1) :
            ret, frame = video_capture.read()
            if not ret:
                break
            if i in frame_ids:
                frames.append(frame)

        while len(frames) < num_frames:
            frames.extend(frames[:num_frames - len(frames)])
        video_capture.release()
        if if_teacher == 1:
            for i in range(len(frames)):
                frames[i] = detector(frames[i])
        frames = [
            preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0) for c in
            frames]
        return frames

    @abstractmethod
    def load_annotations(self, ann_file,data_prefix,num_frames,input_size,preprocess,
                         device, shot, type, if_teacher, detector):
        """Load the annotation according to ann_file into video_infos."""

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        return self.video_info[idx]


class VideoDataset(BaseDataset):
    def __init__(self, config,preprocess,device,ann_file,shot=0,type = 'train'):
        super().__init__(config,preprocess,device,ann_file,shot,type)
        self.labels_file = config.DATA.LABEL_LIST

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self, ann_file,data_prefix,num_frames,input_size,preprocess,
                         device, shot, type, if_teacher, detector):
        """Load annotation file to get video information."""
        video_infos = []
        class_counts = {}
        total_lines = sum(1 for line in open(ann_file, 'r'))
        if type == 'train':
            with open(ann_file, 'r') as fin:
                lines = fin.readlines()
                for idx in range(total_lines):  # Start from the last third
                    if idx % 5 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if label in class_counts and class_counts[label] >= shot:
                        continue
                    data = self.prepare_frames(data_prefix + filename, num_frames, if_teacher, detector, preprocess)
                    if data is not None:
                        video_infos.append(dict(filename=filename, label=label, data=data))
                        if label not in class_counts:
                            class_counts[label] = 1
                        else:
                            class_counts[label] += 1
        elif type == 'test':
            with open(ann_file, 'r') as fin:
                lines = fin.readlines()
                for idx in range(total_lines):  # Start from the last third
                    if idx % 5 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    data = self.prepare_frames(data_prefix + filename, num_frames, if_teacher, detector, preprocess)
                    if data is not None:
                        video_infos.append(dict(filename=filename, label=label, data=data))

        return video_infos



class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    def __len__(self):
        return len(self.indices)
    def set_epoch(self, epoch):
        self.epoch = epoch

def build_dataloader(config,logger):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    _, preprocess = clip.load(config.MODEL.ARCH, device=device)

    test_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TEST_FILE,type='test')
    sampler_test = SubsetRandomSampler(np.arange(len(test_data)))
    test_loader = DataLoader(test_data, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler_test,
                                 num_workers=12, pin_memory=True, drop_last=True)

    logger.info("test_data_finished!")

    train_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=config.DATA.TRAIN_FILE,
                                     shot=config.DATA.SHOTS, type='train')
    sampler_test = SubsetRandomSampler(np.arange(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler_test,
                             num_workers=12, pin_memory=True, drop_last=True)

    return  train_data, test_data, train_loader , test_loader
