import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import model.TClip as tbaclip
import clip
import torch
import os
import argparse
import torch.nn.functional as F
import time
from tqdm import tqdm


from model.transformer import FSATransformerEncoder
from utils.config import get_config
from dataSets.build import build_dataloader
from utils.logger import create_logger
from utils.tools import AverageMeter, classes, visual
import torch.nn as nn
import matplotlib.pyplot as plt
from timm.loss import LabelSmoothingCrossEntropy
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/zero_shot/eval/hmdb/tba_clip_hmdb51_base.yaml')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--shots', type=int)
    parser.add_argument('--temporal_pooling', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--if_teacher', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--zs', type=int)
    parser.add_argument('--lp', type=int)
    parser.add_argument('--label_smooth', type=int)
    args = parser.parse_args()
    config = get_config(args)
    return args, config

@torch.no_grad()
def validate(output, label, plot = False):
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


def train(model,config,train_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, eps=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS * len(train_loader))
    criterion = LabelSmoothingCrossEntropy() if config.MODEL.LABEL_SMOOTH == 1 else nn.CrossEntropyLoss()

    for train_idx in range(config.TRAIN.EPOCHS):
        model.train()
        loss_list = []
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, config.TRAIN.EPOCHS))
        for idx, batch_data in enumerate(tqdm(train_loader)):
            images = batch_data['data']
            images = torch.stack(images)
            images = torch.transpose(images, 0, 1)
            label_id = batch_data['label']
            logits, image_features, text_features = model(images)
            label_id = label_id.to(logits.device)
            loss = criterion(logits, label_id)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Loss: {:.4f}'.format(current_lr,sum(loss_list) / len(loss_list)))

def main(config):
    cache_dir = './caches/' +  config.DATA.DATASET + '/'
    os.makedirs(cache_dir, exist_ok=True)
    config.defrost()  # Unfreeze the config
    config.CACHE_DIR = cache_dir
    config.freeze()  # Freeze the config again
    if not os.path.exists(config.OUTPUT):
        with open(config.OUTPUT, 'w') as f:
            pass
    # Check if the file is empty
    if os.stat(config.OUTPUT).st_size == 0:
        with open(config.OUTPUT, 'a') as f:
            # Write the column names
            f.write('Model,Arch,If_teacher,Num_Frames,Acc1,Acc3,Acc5,AUC,F1,Dataset,Shots,n_ctx,TEMPORAL_POOLING, test_file\n')
    train_data, test_data, train_loader, test_loader = build_dataloader(config, logger)
    class_names = [class_name for i, class_name in classes(config)]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = tbaclip.returnCLIP(config, class_names, device,logger)
    logger.info("Getting textual features as CLIP's classifier.")

    logger.info("Training Begin.")
    if config.MODEL.ZS != 1:
        train(model, config, train_loader)

    logger.info("Eval Model on Test Set.")
    model.eval()
    label_list = []
    logits_list = []
    for idx, batch_data in enumerate(tqdm(test_loader)):
        images = batch_data['data']
        images = torch.stack(images)
        images = torch.transpose(images, 0, 1)
        label_id = batch_data['label']
        logits, image_features, text_features = model(images)
        logits = torch.stack(logits)
        label_id = label_id.to(logits.device)
        logits_list.append(logits)
        label_list.append(label_id)
    logits = torch.cat(logits_list)
    label = torch.cat(label_list)
    acc1, acc3, acc5, auc, f1 = validate(logits, label)
    logger.info(
        "**** Test accuracy1: {:.2f}. , accuracy3: {:.2f},accuracy5: {:.2f}. auc: {:.2f}, f1: {:.2f}****\n".format(
            acc1, acc3, acc5, auc, f1))
    with open(config.OUTPUT, 'a') as f:
        mode = "Zero-shot" if config.MODEL.ZS == 1 else "Few-shot"
        f.write(
            f'{mode},{config.MODEL.ARCH},{config.MODEL.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc3:.3f},{acc5:.3f},{auc:.3f},{f1:.3f},{config.DATA.DATASET},'
            f'{config.DATA.SHOTS} ,{str(config.TEXT_PROMPT.N_CTX_PRE) + " " + str(config.TEXT_PROMPT.N_CTX_POST)},{config.MODEL.TEMPORAL_POOLING}, {config.DATA.TEST_FILE}\n')



if __name__ == '__main__':
    args, config = parse_option()
    if not os.path.exists('train_output'):
        os.makedirs('train_output')
    logger = create_logger(output_dir='train_output', dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(config)

    main(config)