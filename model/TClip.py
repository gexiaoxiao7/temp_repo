import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import model.clip as clip
from collections import OrderedDict
from einops import rearrange

import cv2
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from model.transformer import FSATransformerEncoder

_tokenizer = _Tokenizer()

class VideoEncoder(nn.Module):
    def __init__(self, clip_model, preprocess,config,device):
        super().__init__()
        self.model = clip_model
        self.preprocess = preprocess
        self.dtype = clip_model.dtype
        self.device = device
        self.config = config

        if config.MODEL.TEMPORAL_POOLING == 'fsattention':
            self.attention_net = FSATransformerEncoder(dim=clip_model.visual.output_dim, depth=6,
                                                  heads=1, dim_head=64,
                                                  mlp_dim=clip_model.visual.output_dim * 4,
                                                  dropout=0.1).to(device)
        for name, p in self.model.named_parameters():
            if 'visual.adapter.' not in name:
                p.requires_grad = False

    def forward(self, images):
        video_info = images # b, n_frames, 1 , c, height, weight
        b, n_frames,_,c,h,w = video_info.shape
        video_info = video_info.reshape(-1, c,h,w)
        visual_features, mlp_logits, fused_feats = self.model.encode_image(video_info.to(self.device).to(self.dtype))
        # b*n_frames, dim -> b, n_frames, dim
        visual_features = visual_features.reshape(b,n_frames, -1)
        mlp_logits = mlp_logits.reshape(b,n_frames, -1)
        fused_feats = fused_feats.reshape(b,n_frames, -1)
        # b, n_frames, dim
        if self.config.MODEL.TEMPORAL_POOLING == 'fsattention':
            visual_features= self.attention_net(visual_features)
            with torch.no_grad():
                fused_feats = self.attention_net(fused_feats)
        visual_features = torch.mean(visual_features, dim=1)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        mlp_logits = torch.mean(mlp_logits, dim=1)
        mlp_logits = mlp_logits / mlp_logits.norm(dim=-1, keepdim=True)
        fused_feats = torch.mean(fused_feats, dim=1)
        fused_feats = fused_feats / fused_feats.norm(dim=-1, keepdim=True)
        return visual_features, mlp_logits, fused_feats



class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,device,logger):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pre = cfg.TEXT_PROMPT.N_CTX_PRE
        n_ctx_post = cfg.TEXT_PROMPT.N_CTX_POST
        ctx_pre_init = cfg.TEXT_PROMPT.CTX_PRE_INIT
        ctx_post_init = cfg.TEXT_PROMPT.CTX_POST_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        if ctx_pre_init:
            # use given words to initialize context vectors
            ctx_pre_init = ctx_pre_init.replace("_", " ")
            n_ctx_pre = len(ctx_pre_init.split(" "))
            prompt = clip.tokenize(ctx_pre_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_pre_vectors = embedding[0, 1: 1 + n_ctx_pre, :]
            prompt_prefix = ctx_pre_init
        else:
            # random initialization
            ctx_pre_vectors = torch.empty(n_ctx_pre, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_pre_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx_pre)

        if ctx_post_init:
            ctx_post_init = ctx_post_init.replace("_", " ")
            n_ctx_post = len(ctx_post_init.split(" "))
            prompt = clip.tokenize(ctx_post_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_post_vectors = embedding[0, 1: 1 + n_ctx_post, :]
            prompt_suffix = ctx_post_init
        else:
            ctx_post_vectors = torch.empty(n_ctx_post, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_post_vectors, std=0.02)
            prompt_suffix = " ".join(["X"] * n_ctx_post)

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx_pre}")
        logger.info(f'Final context: "{prompt_suffix}"')
        logger.info(f"Number of context words (tokens): {n_ctx_post}")


        self.ctx_pre = nn.Parameter(ctx_pre_vectors)
        self.ctx_post = nn.Parameter(ctx_post_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=False)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ])).to(device)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + " " + prompt_suffix for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p).to(device) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_cls", embedding[:, 1 + n_ctx_pre: -1 - n_ctx_post , :])  # CLS
        self.register_buffer("token_suffix", embedding[:, -1:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx_pre = n_ctx_pre
        self.n_ctx_post = n_ctx_post
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.device = device

    def construct_prompts(self, pre_ctx, cls, post_ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                pre_ctx,  # (dim0, n_ctx, dim)
                cls,  # (dim0, n_cls, dim)
                post_ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        im_features = im_features.reshape(-1, im_features.shape[-1])
        prefix = self.token_prefix
        cls = self.token_cls
        suffix = self.token_suffix
        ctx_pre = self.ctx_pre  # (n_ctx, ctx_dim)
        ctx_post = self.ctx_post  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx_pre = ctx_pre.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_post = ctx_post.unsqueeze(0)
        ctx_shifted_pre = ctx_pre.to(self.device) + bias.to(self.device)  # (batch, n_ctx, ctx_dim)
        ctx_shifted_post = ctx_post.to(self.device) + bias.to(self.device)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i_pre,ctx_shifted_i_post in zip(ctx_shifted_pre, ctx_shifted_post):
            ctx_i_pre = ctx_shifted_i_pre.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_i_post = ctx_shifted_i_post.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i_pre, cls, ctx_i_post, prefix,suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts # [batch, n_cls, n_tkn, ctx_dim]

class TextEncoder(nn.Module):
    def __init__(self, clip_model, config, classnames, device, logger):
        super().__init__()
        self.model = clip_model
        self.classnames = classnames
        self.config = config
        self.device = device
        # use prompt learner
        if config.MODEL.LP == 1:
            self.prompts_learner = PromptLearner(config, classnames, clip_model,device,logger)
            self.tokenized_prompts = self.prompts_learner.tokenized_prompts

        for name, p in self.model.named_parameters():
            if 'visual.adapter.' not in name:
                p.requires_grad = False

    def forward(self, im_features):

        if self.config.MODEL.LP == 1:
            logit_scale = self.model.logit_scale.exp()
            prompts = self.prompts_learner(im_features) # (b, n_cls, n_tkn, ctx_dim)
            logits = []
            for pts_i, imf_i in zip(prompts, im_features):
                text_features = self._forward(pts_i, self.tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = (logit_scale * imf_i @ text_features.t()).softmax(dim=-1)
                logits.append(l_i)
            logits = torch.stack(logits)
        else:
            prompts = ['The person was ' + x + '.' for x in self.classnames]
            x = [clip.tokenize(prompt).to(self.device) for prompt in prompts]
            clip_weights = [self.model.encode_text(i) for i in x]
            clip_weights = torch.stack(clip_weights)
            clip_weights = clip_weights.squeeze(dim=1)
            clip_weights /= clip_weights.norm(dim=-1, keepdim=True)
            text_features = clip_weights
            norm = text_features.norm(dim=-1, keepdim=True)
            text_feature = text_features / norm
            logits = (100.0 * im_features @ text_feature.T).softmax(dim=-1)

        return logits


    def _forward(self, prompts, tokenized_prompts=None):
        x = prompts + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.model.text_projection

        return x


class TBA_Clip(nn.Module):
    def __init__(self, clip_model, preprocess, classnames, device, logger ,config):
        super().__init__()
        self.model = clip_model
        self.preprocess = preprocess
        self.text_encoder = TextEncoder(clip_model,config,classnames,device,logger)
        self.image_encoder = VideoEncoder(clip_model, preprocess, config,device)
        self.dtype = clip_model.dtype
        self.config = config
        self.device = device
        self.classnames = classnames

        feature_dim = clip_model.visual.output_dim
        num_classes = config.DATA.NUM_CLASSES
        self.adapter = nn.ModuleDict({
            "adapter_1": nn.Sequential(nn.Linear(num_classes, feature_dim, bias=False), nn.BatchNorm1d(feature_dim),
                                       nn.ReLU(inplace=False), nn.Linear(feature_dim, feature_dim, bias=False)),
            "adapter_2": nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False), nn.BatchNorm1d(feature_dim),
                                       nn.ReLU(inplace=False), nn.Linear(feature_dim, feature_dim, bias=False)),
            "adapter_3": nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False), nn.BatchNorm1d(feature_dim),
                                       nn.ReLU(inplace=False), nn.Linear(feature_dim, num_classes, bias=False)),
            "g_weight": nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False), nn.BatchNorm1d(feature_dim),
                                      nn.ReLU(inplace=False), nn.Linear(feature_dim, 1, bias=False), nn.Sigmoid()),
        })

        for name, p in self.model.named_parameters():
            if 'visual.adapter.' not in name:
                p.requires_grad = False




    def forward(self, image):
        visual_feats, mlp_logits, fused_feats = self.image_encoder(image) #(b, 1,dim)
        clip_logits = self.text_encoder(visual_feats) # (b, n_cls, dim)
        ada_logits = clip_logits + self.adapter["adapter_3"](
            self.adapter["adapter_1"](clip_logits) + self.adapter["adapter_2"](fused_feats))
        weight = self.adapter["g_weight"](fused_feats)  # logits weight
        total_logits = weight * mlp_logits + (1 - weight) * ada_logits
        return clip_logits, mlp_logits, ada_logits, total_logits




def returnCLIP(config,classnames,device,logger):
    clip_model, preprocess = clip.load(config.MODEL.ARCH, device = device, config=config)
    model = TBA_Clip(clip_model, preprocess,classnames,device,logger,config).to(device)
    return model


