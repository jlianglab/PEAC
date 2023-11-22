import os

import yaml
from yacs.config import CfgNode as CN

_C = CN() # 创建一个CN容器来装载参数
# mode: train or test
_C.LOCAL_RANK = ''
_C.TASK = ''

_C.DATA = CN()
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 8
_C.DATA.IMG_SIZE = 448
_C.DATA.PATCH_SIZE = 4
_C.DATA.DATA_PATH = '/data/zhouziyu/ssl/NIHChestX-ray14/images/'
_C.DATA.TRAIN_LIST = '/home/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/Xray14_train_official.txt'
_C.DATA.VAL_LIST = '/home/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/Xray14_val_official.txt'
_C.DATA.TEST_LIST = '/home/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/Xray14_test_official.txt'


_C.TRAIN = CN()
_C.TRAIN.GPU = ''
_C.TRAIN.LR = 0.1
_C.TRAIN.EPOCHS = 400
_C.TRAIN.TEACHER_M = 0.999
_C.TRAIN.ABLATION_MODE = 'popar'


_C.MODEL = CN()
_C.MODEL.WEIGHT = ''
_C.MODEL.DEPTH = '2,2,18,2'
_C.MODEL.HEADS = '4,8,16,32'
_C.MODEL.MLP = '8192-8192-8192'
_C.MODEL.MLPLOCAL = '512-512-512'
_C.MODEL.OUTPUT = ''


def update_config(config, args):

    def _check_args(name):
        # if hasattr(args, name) and eval(f'args.{name}'): # hasattr判断args中是否有name属性
        if hasattr(args, name):    
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('img_size'):
        config.DATA.IMG_SIZE  = args.img_size
    if _check_args('patch_size'):
        config.DATA.PATCH_SIZE = args.patch_size
    if _check_args('gpu'):
        config.TRAIN.GPU = args.gpu
    if _check_args('lr'):
        config.TRAIN.LR = args.lr
    if _check_args('epochs'):
        config.TRAIN.EPOCHS = args.epochs
    if _check_args('teacher_m'):
        config.TRAIN.TEACHER_M = args.teacher_m
    if _check_args('weight'):
        config.MODEL.WEIGHT = args.weight
    if _check_args('depth'):
        config.MODEL.DEPTH = args.depth
    if _check_args('heads'):
        config.MODEL.HEADS = args.heads
    if _check_args('mlp'):
        config.MODEL.MLP = args.mlp
    if _check_args('ablation_mode'):
        config.TRAIN.ABLATION_MODE = args.ablation_mode
    if _check_args('task'):
        config.TASK = args.task
    if _check_args('output'):
        config.MODEL.OUTPUT = os.path.join(args.output, "pretrained_weight", config.TRAIN.ABLATION_MODE+'_'+config.TASK)
    if _check_args('local_rank'):
        config.LOCAL_RANK = args.local_rank
    


class get_config:
    def __init__(self, args):
        self.config = _C.clone()
        update_config(self.config, args)

        if not os.path.exists(self.config.MODEL.OUTPUT):
            os.mkdir(self.config.MODEL.OUTPUT)
        logs_path = os.path.join(self.config.MODEL.OUTPUT, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if os.path.exists(os.path.join(logs_path, "log.txt")):
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:", file=self.log_writter)
        print(self.config.dump(), file=self.log_writter)
        print("\n", file=self.log_writter)
