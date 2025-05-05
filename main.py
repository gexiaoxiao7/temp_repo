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


from model.tp import Attention
from model.transformer import FSATransformerEncoder
from utils.config import get_config
from dataSets.build import build_dataloader
from utils.logger import create_logger
from utils.tools import AverageMeter, clip_classifier, build_cache_model, pre_load_features, split_dataset, \
    attention_Fuc, promptlearner_Fuc, classes, visual
from utils.tools import  search_hp
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

def train_lp(clip_model,device,config,train_loader,class_names,attention_net, test_features, attention_test_feature,test_labels):

    if config.MODEL.TEMPORAL_POOLING == 'attention':
        attention_net.load_state_dict(
            torch.load(config.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "attention_model.pth"))

    prompt_learner = tbaclip.PromptLearner(config, class_names, clip_model.model, device,logger).to(torch.half)

    if os.path.exists(config.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "prompt_learner.pth"):
        prompt_learner.load_state_dict(torch.load(config.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "prompt_learner.pth"))

    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=config.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS * len(train_loader))
    criterion = LabelSmoothingCrossEntropy() if config.MODEL.LABEL_SMOOTH == 1 else nn.CrossEntropyLoss()
    for train_idx in range(config.TRAIN.EPOCHS):
        # Train
        b = config.TRAIN.BATCH_SIZE
        prompt_learner.train()
        loss_list = []
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, config.TRAIN.EPOCHS))
        for idx, batch_data in enumerate(tqdm(train_loader)):
            images = batch_data['data']
            images = torch.stack(images)
            images = torch.transpose(images, 0, 1)
            label_id = batch_data['label']
            with torch.no_grad():
                _, image_features, _,attention_format = clip_model(images)
                if config.MODEL.TEMPORAL_POOLING == 'attention':
                    image_features = attention_Fuc(attention_net, attention_format, image_features)
                else:
                    image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = []
            prompts = prompt_learner(image_features)
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = clip_model.text_encoder(pts_i, prompt_learner.tokenized_prompts)
                # text_features = clip_model.module.text_encoder(pts_i, prompt_learner.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = (clip_model.logit_scale.exp() * imf_i @ text_features.t()).softmax(dim=-1)
                logits.append(l_i)
            logits = torch.stack(logits)
            # 修改成label_smooth的损失函数
            label_id = label_id.to(logits.device)
            loss = criterion(logits, label_id)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            prompt_learner.float()
            optimizer.step()
            prompt_learner.half()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Loss: {:.4f}'.format(current_lr,sum(loss_list) / len(loss_list)))

        torch.save(prompt_learner.state_dict(),
                   config.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "prompt_learner.pth")

    # Eval
    video_feature = attention_Fuc(attention_net, attention_test_feature, test_features)
    logits = promptlearner_Fuc(prompt_learner, video_feature, clip_model)
    test_labels = test_labels.to(logits.device)
    acc1, acc3, acc5, auc, f1 = validate(logits, test_labels)
    logger.info(
        "**** Test accuracy1: {:.2f}, accuracy3: {:.2f},accuracy5: {:.2f}. auc: {:.2f}, f1: {:.2f}****\n".format(
            acc1, acc3, acc5, auc, f1))
    return prompt_learner

def train_attention(clip_model,device,config,train_loader,clip_weights,test_features, attention_test_feature,test_labels):
    attention_net = FSATransformerEncoder(dim=clip_model.model.visual.output_dim, depth=6,
                                      heads=1, dim_head=64,
                                      mlp_dim=clip_model.model.visual.output_dim * 4, nt=config.DATA.NUM_FRAMES,
                                      nh=1, nw=1,
                                      dropout=0.1).to(device).to(torch.half)
    if os.path.exists(config.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "attention_model.pth"):
        attention_net.load_state_dict(torch.load(config.CACHE_DIR + "/" + str(config.DATA.SHOTS) + "attention_model.pth"))
    optimizer = torch.optim.Adam(attention_net.parameters(), lr=config.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS * len(train_loader))
    criterion = LabelSmoothingCrossEntropy() if config.MODEL.LABEL_SMOOTH == 1 else nn.CrossEntropyLoss()


    for train_idx in range(config.TRAIN.EPOCHS):
        # Train
        b = config.TRAIN.BATCH_SIZE
        attention_net.train()
        loss_list = []
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, config.TRAIN.EPOCHS))
        for idx, batch_data in enumerate(tqdm(train_loader)):
            images = batch_data['data']
            images = torch.stack(images)
            images = torch.transpose(images, 0, 1)
            label_id = batch_data['label']
            with torch.no_grad():
                _,image_features,_,image_features_attention = clip_model(images)
            attention_weights = attention_net(image_features_attention)

            weighted_features = torch.mul(attention_weights, image_features)
            video_feature = torch.mean(weighted_features, dim=1)

            norm = video_feature.norm(dim=-1, keepdim=True)
            video_features = video_feature / norm
            clip_logits = (100. * video_features @ clip_weights.T).softmax(dim=-1)

            label_id = label_id.to(clip_logits.device)
            loss = criterion(clip_logits, label_id)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Loss: {:.4f}'.format(current_lr,sum(loss_list) / len(loss_list)))

        torch.save(attention_net.state_dict(), config.CACHE_DIR + "/" + str(config.DATA.SHOTS) +"attention_model.pth")

    # Eval
    video_features = attention_Fuc(attention_net, attention_test_feature, test_features)
    clip_logits = (100. * video_features @ clip_weights.T).softmax(dim=-1)
    test_labels = test_labels.to(clip_logits.device)
    # logger.info(f"clip_logits shape:{clip_logits.shape}\n clip_logits:{clip_logits}")
    # logger.info(f"test_labels shape:{test_labels.shape}\n test_labels:{test_labels}")
    acc1, acc3, acc5, auc, f1 = validate(clip_logits, test_labels)
    logger.info(
        "**** Test accuracy1: {:.2f}. , accuracy3: {:.2f},accuracy5: {:.2f}. auc: {:.2f}, f1: {:.2f}****\n".format(
            acc1, acc3, acc5, auc, f1))

    return attention_net

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
            f.write('Model,Arch,If_teacher,Num_Frames,Acc1,Acc3,Acc5,AUC,F1,Dataset,Shots,n_ctx,cache_size,TEMPORAL_POOLING, test_file\n')
    (train_cache_data, val_data, test_data,train_data_F, train_data_a,
     train_load_cache, val_loader, test_loader, train_load_F, train_load_a)= build_dataloader(config, logger)
    class_names = [class_name for i, class_name in classes(config)]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = tbaclip.returnCLIP(config, class_names, device,logger)
    logger.info("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(class_names, model, config,device)
    # Pre-load val features
    logger.info("Loading visual features and labels from val set.")
    val_features, val_labels,attention_val_feature = pre_load_features(config, "val", model, val_loader)
    # Pre-load test features
    logger.info("Loading visual features and labels from test set.")
    test_features, test_labels, attention_test_feature = pre_load_features(config, "test", model, test_loader)
    logger.info("Training attention Net.")
    if config.MODEL.TEMPORAL_POOLING == 'attention' and config.MODEL.ZS == 0:
        attention_net = train_attention(model, device, config, train_load_F, clip_weights, test_features, attention_test_feature, test_labels)
    # use prompt_learner
    logger.info("Training Prompt Learner.")
    if config.MODEL.LP == 1 and config.MODEL.ZS == 0:
        prompt_learner = train_lp(model, device, config, train_load_a, class_names,attention_net, test_features, attention_test_feature, test_labels)

    if config.MODEL.ZS == 1:
        clip_logits = (100. * val_features @ clip_weights.T).softmax(dim=-1)
        acc1, acc3, acc5, auc, f1 = validate(clip_logits, val_labels)
        logger.info(
            "\n**** Zero-shot CLIP's val accuracy1: {:.2f}. accuracy3: {:.2f} accuracy5: {:.2f} auc:{:.2f} f1:{:.2f} ****\n"
            .format(acc1, acc3, acc5, auc, f1))
        with open(config.OUTPUT, 'a') as f:
            f.write(
                f'Zero-shot,{config.MODEL.ARCH},{config.MODEL.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc3:.3f},{acc5:.3f},{auc:.3f},{f1:.3f},{config.DATA.DATASET},'
                f'0 ,{str(config.TEXT_PROMPT.N_CTX_PRE) + " " + str(config.TEXT_PROMPT.N_CTX_POST)},{config.DATA.CACHE_SIZE},{config.MODEL.TEMPORAL_POOLING}, {config.DATA.TEST_FILE}\n')
    else:
        if config.MODEL.TEMPORAL_POOLING == 'attention':
            test_features = test_features.squeeze(1) if config.TEMPORAL_POOLING != 'attention' else attention_Fuc(
                attention_net,
                attention_test_feature, test_features)
            val_features = val_features.squeeze(1) if config.TEMPORAL_POOLING != 'attention' else attention_Fuc(
                attention_net, attention_val_feature,
                val_features)
        clip_logits = 100. * test_features @ clip_weights.T if config.TRAIN.LP == 0 else promptlearner_Fuc(
            prompt_learner, test_features, model)
        acc1, acc3, acc5, auc, f1 = validate(clip_logits, test_labels)
        logger.info(
            "**** Tip-Adapter-F's test accuracy1: {:.2f}. , accuracy3: {:.2f},accuracy5: {:.2f}. auc: {:.2f}, f1: {:.2f}****\n".format(
                acc1, acc3, acc5, auc, f1))
        with open(config.OUTPUT, 'a') as f:
            f.write(
                f'Tip-Adapter-F,{config.MODEL.ARCH},{config.MODEL.IF_TEACHER},{config.DATA.NUM_FRAMES},{acc1:.3f},{acc3:.3f},{acc5:.3f},{auc:.3f},{f1:.3f},{config.DATA.DATASET},'
                f'{config.DATA.SHOTS} ,{str(config.TEXT_PROMPT.N_CTX_PRE) + " " + str(config.TEXT_PROMPT.N_CTX_POST)},{config.DATA.CACHE_SIZE},{config.TEMPORAL_POOLING}, {config.DATA.TEST_FILE}\n')



if __name__ == '__main__':
    args, config = parse_option()
    if not os.path.exists('train_output'):
        os.makedirs('train_output')
    logger = create_logger(output_dir='train_output', dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(config)

    main(config)