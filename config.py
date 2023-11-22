import os
import sys

import torch

DATA_PATH = '/data/zhouziyu/ssl/NIHChestX-ray14/images/'

class ChestX_ray14:

    server = "lab"  # server = lab | psc | agave
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.task = args.task
        self.dataset = args.dataset
        self.test = args.test
        self.weight = args.weight
        self.gpu = args.gpu
        self.depth = args.depth
        self.heads = args.heads
        self.in_channel = args.in_channel
        self.patch_size = 4
        self.path = args.path
        self.local_rank = args.local_rank
        self.ablation_mode = args.ablation_mode

        self.method = self.task+"_depth"+str(self.depth)+ \
                      "_head" + str(self.heads)+"_" + \
                      self.dataset + "_in_channel" + str(self.in_channel)

        if self.dataset == "nih14":
            self.train_image_path_file = [
                (DATA_PATH, "/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/Xray14_train_official.txt"),
                (DATA_PATH, "/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/Xray14_val_official.txt"),
            ]
            self.test_image_path_file = [
                (DATA_PATH, "/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/Xray14_test_official.txt"),
            ]

        # self.model_path = os.path.join("pretrained_weight", self.method )
        self.model_path = os.path.join(self.path, "popar_pretrained_weight", self.ablation_mode, self.method )

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if os.path.exists(os.path.join(logs_path, "log.txt")):
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

        self.graph_path = os.path.join(logs_path, "graph_path")
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        if args.gpu is not None:
            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.debug_mode:
            self.log_writter = sys.stdout

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:", file=self.log_writter)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
        print("\n", file=self.log_writter)