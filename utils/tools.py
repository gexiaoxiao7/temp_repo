import torch.distributed as dist
import torch
import clip
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score, f1_score
import cv2
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import clip

from model.TClip import load_clip

matplotlib.use('Agg')
from model.tld import TeacherDetection
import matplotlib.pyplot as plt
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def classes(config):
    print(config.DATA.LABEL_LIST)
    classes_all = pd.read_csv(config.DATA.LABEL_LIST)
    return classes_all.values.tolist()



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def validate(output, label, plot = False, config = None):
    acc1_meter, acc5_meter,acc3_meter = AverageMeter(), AverageMeter(), AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = label.clone().detach().to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    for idx, similarity in enumerate(output):
        cur_label = label[idx]
        value1, indices_1 = similarity.topk(1, dim=-1)
        value3, indices_3 = similarity.topk(3, dim=-1)
        value5, indices_5 = similarity.topk(5, dim=-1)
        acc1, acc3 ,acc5 = 0, 0,0
        for i in range(1): # batch_size
            if indices_1[i] == cur_label:
                acc1 += 1
            if cur_label in indices_3:
                acc3 += 1
            if cur_label in indices_5:
                acc5 += 1
        acc1_meter.update(float(acc1) * 100,1)
        acc3_meter.update(float(acc3) * 100, 1)
        acc5_meter.update(float(acc5) * 100,1)
        all_preds.append(indices_1.cpu().numpy())
        all_labels.append(cur_label.cpu().numpy())
        probs = similarity.softmax(dim=-1).cpu().detach().numpy()
        if len(probs.shape) > 1:  # 如果probs有多个维度
            probs /= probs.sum(axis=1, keepdims=True)  # 归一化概率，使其和为1
        else:  # 如果probs只有一个维度
            probs /= probs.sum()  # 归一化概率，使其和为1
        if not np.isclose(probs.sum(), 1):
            probs = np.clip(probs, 0, 1)
            min_index = np.argmin(probs)
            sum = 0
            for i,num in enumerate(probs):
                if i != min_index:
                    sum += num
            probs[min_index] = 1 - sum
        all_probs.append(probs)
    # AUC and F1
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    f1 = f1_score(all_labels, all_preds, average='macro')
    if plot:
        cls = classes(config)
        labels = [sublist[1] for sublist in cls]

        cm = confusion_matrix(np.array(all_labels), np.array(all_preds))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Convert to percentages

        fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, shrink=0.7)  # Adjust the length of colorbar

        # Show all ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'),  # Show 2 decimal places
                        ha='center', va='center', color='black')

        fig.tight_layout()  # Increase margin
        plt.savefig('confusion_matrix.png')

    return acc1_meter.avg, acc3_meter.avg, acc5_meter.avg, auc, f1

def split_dataset(dataset):
    # Step 1: Create a list of indices for each label
    label_to_indices = defaultdict(list)
    for idx, batch_data in enumerate(dataset):
        label = batch_data['label']
        label_to_indices[label].append(idx)

    # Step 2: Shuffle and split the indices for each label and add them to the new index lists
    indices1, indices2 = [], []
    for indices in label_to_indices.values():
        random.shuffle(indices)  # Shuffle the indices
        mid = len(indices) // 2
        if len(indices) % 2 == 1:  # Check if the number of samples is odd
            indices1.extend(indices[:mid+1])  # If odd, subset1 gets one more sample
            indices2.extend(indices[mid+1:])  # subset2 gets one less sample
        else:
            indices1.extend(indices[:mid])
            indices2.extend(indices[mid:])

    # Step 3: Create two Subset objects and two DataLoaders
    subset1 = Subset(dataset, indices1)
    subset2 = Subset(dataset, indices2)

    return subset1,subset2

def visulize_attention_ratio(img_path, attention_mask, ratio=0.5, cmap="jet"):
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)


def prepare_frames(path,num_frames,device):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return None
    video_capture = cv2.VideoCapture(path)
    model , preprocess = load_clip('ViT-L/14@336px',device)
    detector = TeacherDetection('Yolo-model/yolov8n.pt')
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_ids = np.linspace(0, total_frames - 2, num_frames)
    frame_ids = np.floor(frame_ids).astype(int)
    for i in range(total_frames+1) :
        ret, frame = video_capture.read()
        if not ret:
            break
        if i in frame_ids:
            # Resize the frame
            frame = cv2.resize(frame, (224*8, 224*8))
            frames.append(frame)
    while len(frames) < num_frames:
        frames.extend(frames[:num_frames - len(frames)])
    video_capture.release()
    for i in range(len(frames)):
        frames[i] = detector(frames[i])
    return frames
def visual(config, path ,logits):
    # Step 1: Prepare frames from the video
    logits = logits.flatten()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = prepare_frames(path, config.DATA.NUM_FRAMES,device)
    top1_classes = logits.argsort()[-1].item()  # Use .item() to get the value
    top1_probs = logits[top1_classes]
    # Step 2: Add text to each frame and save
    for i, frame in enumerate(frames):
        # Get top 1 class and its probability
        cls = classes(config)
        class_name = cls[top1_classes][1]
        prob = top1_probs
        # Add text to the frame
        text_prob = f'{prob:.2f}:'
        text_class = f'{class_name}'
        cv2.putText(frame, text_prob, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, text_class, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # Save the frame
        # 取path的最后一级路径名
        video_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'output/{video_name}_frame_{i}.jpg', frame)
    for i, frame in enumerate(frames):
        video_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(f'output/{video_name}_frame_orign_{i}.jpg', frame)