import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# Base config files
_C.BASE = ['']
_C.CACHE_DIR = ''
# a csv file to save the results
_C.OUTPUT = 'output.csv'

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.TEST_FILE = ''
_C.DATA.DATASET = 'hmdb51'
_C.DATA.NUM_FRAMES = 30
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_CLASSES = 51
_C.DATA.LABEL_LIST = 'labels/hmdb51_org_base_labels.csv'
# Few-shot setting, k-shot
_C.DATA.SHOTS = 2
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.YOLO = 'Yolo-model/yolov8n.pt'
# If using the teacher detection model
_C.MODEL.IF_TEACHER = 1
# If using zero-shot mode
_C.MODEL.ZS = 0
# if using label smoothing
_C.MODEL.LABEL_SMOOTH = 1
# temporal pooling method: 'mean', 'attention'
_C.MODEL.TEMPORAL_POOLING = 'attention'
# if using prompt learner
_C.MODEL.LP = 0
_C.MODEL.ADAPTER_FUSE_TYPE = 2
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = 0
_C.TRAIN.USE_CHECKPOINT = 0
# -----------------------------------------------------------------------------
# Text Prompt Settings
# -----------------------------------------------------------------------------
_C.TEXT_PROMPT = CN()
_C.TEXT_PROMPT.N_CTX_PRE = 9
_C.TEXT_PROMPT.CTX_PRE_INIT = "The action of the person in the picture is"
_C.TEXT_PROMPT.N_CTX_POST = 3
_C.TEXT_PROMPT.CTX_POST_INIT = "in the classroom"

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if hasattr(args, 'if_teacher') and args.if_teacher is not None:
        config.MODEL.IF_TEACHER = args.if_teacher
    if hasattr(args, 'num_frames') and args.num_frames is not None:
        config.DATA.NUM_FRAMES = args.num_frames
    if hasattr(args, 'arch') and args.arch is not None:
        config.MODEL.ARCH = args.arch
    if hasattr(args, 'temporal_pooling') and args.temporal_pooling is not None:
        config.MODEL.TEMPORAL_POOLING = args.temporal_pooling
    if hasattr(args, 'test_file') and args.test_file is not None:
        config.DATA.TEST_FILE = args.test_file
    if hasattr(args, 'output') and args.output is not None:
        config.OUTPUT = args.output
    if hasattr(args, 'zs') and args.zs is not None:
        config.MODEL.ZS = args.zs
    if hasattr(args, 'shots') and args.shots is not None:
        config.DATA.SHOTS = args.shots
    if hasattr(args, 'lp') and args.lp is not None:
        config.MODEL.LP = args.lp
    if hasattr(args, 'label_smooth') and args.label_smooth is not None:
        config.MODEL.LABEL_SMOOTH = args.label_smooth
    # set local rank for distributed training
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config