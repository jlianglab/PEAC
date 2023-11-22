# popar+global
import argparse
import math
import os
import sys
import time

import config
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from datasets import Popar_chestxray, build_md_transform
from einops import rearrange
from swin_transformer import SwinTransformer
from timm.utils import ModelEma, NativeScaler, get_state_dict
from torch import optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.build_loader import build_loader_NIHchest
from utils.config import get_config
from utils.utils_pec import AverageMeter, cosine_scheduler, save_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16,  help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--gpu', dest='gpu', default="3,6", type=str, help="gpu index")
    parser.add_argument('--task', dest='task', default="POPAR_swin", type=str)
    parser.add_argument('--dataset', dest='dataset', default="nih14", type=str)
    parser.add_argument('--weight', dest='weight', default=None)
    parser.add_argument('--depth', dest='depth', type=str, default="2,2,18,2")
    parser.add_argument('--heads', dest='heads', type=str, default="4,8,16,32")
    parser.add_argument("--mlp", default="8192-8192-8192")
    parser.add_argument('--in_channel', dest='in_channel', default=3, type=int, help="input color channel")
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--teacher_m', type=float, default=0.999, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    # args = parser.parse_args()
    args = parser.parse_args([])
    get_cfg = get_config(args)
    config = get_cfg.config
    return args, config, get_cfg


# device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def step_decay(step, conf,warmup_epochs = 5):
    lr = conf.TRAIN.LR
    progress = (step - warmup_epochs) / float(conf.TRAIN.EPOCHS - warmup_epochs)
    progress = np.clip(progress, 0.0, 1.0)
    #decay_type == 'cosine':
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    if warmup_epochs:
      lr = lr * np.minimum(1., step / warmup_epochs)
    return lr


class _SwinTransformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.num_classes == 0

    def forward(self, x):
        x = self.patch_embed(x)

        B, L, _ = x.shape
        if self.ape: # absolute position embedding
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x) # Dropout

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x) # Layer Normalization

        # x = x.transpose(1, 2)
        # B, C, L = x.shape
        # H = W = int(L ** 0.5)
        # x = x.reshape(B, C, H, W)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SwinModel(nn.Module):
    def __init__(self,hidden_size = 128, num_classes = 196, depth=[ 2, 2, 18, 2 ],heads=[ 4, 8, 16, 32 ]):
        super(SwinModel, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.depth = depth
        self.heads = heads
        self.swin_model = _SwinTransformer(img_size=448,patch_size=4,in_chans=3,num_classes=0,embed_dim=self.hidden_size,depths=self.depth,num_heads= self.heads,
                                          window_size=7,mlp_ratio=4.,qkv_bias=True,qk_scale=None,drop_rate=0,drop_path_rate=0.1,ape=False,patch_norm=True,use_checkpoint=False)

        self.head = nn.Linear(1024 , self.num_classes,bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.head.bias = self.bias
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=32 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(32),
            # nn.Conv2d(3, 3, kernel_size=1),
            # nn.BatchNorm2d(3),
            # nn.GELU(),
            # nn.Conv2d(3, 3, kernel_size=1),
            # nn.BatchNorm2d(3),
            # nn.GELU(),
        )

    def forward(self, img_x,perm):
        B,C,H,W = img_x.shape

        img_x = rearrange(img_x, 'b c (h p1) (w p2)-> b (h w) c p1 p2', p1=32, p2=32, w=14,h=14) # 切分后patch大小为32*32，patch个数14*14
        for i in range(B):
            img_x[i] = img_x[i,perm[i],:,:,:]
        img_x = rearrange(img_x, 'b (h w) c p1 p2 -> b c (h p1) (w p2)', p1=32, p2=32, w=14,h=14)
        out = self.swin_model(img_x)
        B, L, C = out.shape # img_size=448, out.shape=[B, 196, 1024]

        cls_feature = out.reshape(-1, 1024)
        restor_feature = out.transpose(1, 2)
        H = W = int(L ** 0.5)
        restor_feature = restor_feature.reshape(B, C, H, W)

        decoder_out = self.decoder(restor_feature)

        return decoder_out, self.head(cls_feature)
    
class PEC_Model(nn.Module):
    def __init__(self, config, hidden_size = 128, num_classes = 196, depth=[ 2, 2, 18, 2 ],heads=[ 4, 8, 16, 32 ]):
        super(PEC_Model, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size*8 # encoder输出层维度
        self.depth = depth
        self.heads = heads
        self.swin_model = _SwinTransformer(img_size=448,patch_size=4,in_chans=3,num_classes=0,embed_dim=self.hidden_size,depths=self.depth,num_heads= self.heads,
                                          window_size=7,mlp_ratio=4.,qkv_bias=True,qk_scale=None,drop_rate=0,drop_path_rate=0.1,ape=False,patch_norm=True,use_checkpoint=False)
        self.mlp = self.MLP(config.MODEL.MLP, self.embed_dim)

        # popar分类和复原头
        self.head = nn.Linear(1024 , self.num_classes,bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.head.bias = self.bias

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=32 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(32)
        )

    def forward(self, img_x, perm): # img_x用于popar输入
        B,C,H,W = img_x.shape

        img_x = rearrange(img_x, 'b c (h p1) (w p2)-> b (h w) c p1 p2', p1=32, p2=32, w=14,h=14) # 切分后patch大小为32*32，patch个数14*14
        for i in range(B):
            img_x[i] = img_x[i,perm[i],:,:,:] # perm:[80,196]，其中有1/3的概率patch是打乱顺序的
        img_x = rearrange(img_x, 'b (h w) c p1 p2 -> b c (h p1) (w p2)', p1=32, p2=32, w=14,h=14) # img_x: [80,3,448,448]

        out = self.swin_model(img_x) # out B,H*W,C (80,196,1024)
        B, L, C = out.shape

        cls_feature = out.reshape(-1, 1024)
        order_out = self.head(cls_feature)

        restor_feature = out.transpose(1, 2)
        H = W = int(L ** 0.5)
        restor_feature = restor_feature.reshape(B, C, H, W)
        decoder_out = self.decoder(restor_feature)

        avg_out = out.mean(dim=1)
        global_embd = self.mlp(avg_out)


        return order_out, decoder_out, global_embd
    
    def MLP(self, mlp, embed_dim):
        mlp_spec = f"{embed_dim}-{mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.LayerNorm(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)
        

def build_model(conf):
    start_epoch = 1

    if conf.MODEL.WEIGHT is None:
        model = PEC_Model(config=conf)
    else:
        print("Loading pretrained weights", file=conf.log_writter)
        if conf.weight is None:
            model = PEC_Model(config=conf)
            checkpoint = torch.load(os.path.join(conf.model_path, "last.pth"), map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict)
        else:
            model = PEC_Model(config=conf)
            checkpoint = torch.load(conf.weight, map_location='cpu')
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict)
            start_epoch = checkpoint['epoch'] + 1

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr)
    optimizer = optim.SGD(model.parameters(), lr=conf.TRAIN.LR, weight_decay=0, momentum=0.9, nesterov=False)
    # model = model.double() # 把模型参数从float转为double

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True

    loss_scaler = NativeScaler()

    return model, optimizer,loss_scaler,start_epoch


def train(train_loader, student, teacher, momentum_schedule, optimizer, epoch, loss_scaler, conf, writer, log_writer):
    """one epoch training"""
    student.train(True)

    batch_time = AverageMeter()
    losses = AverageMeter()
    global_losses = AverageMeter()
    restor_losses = AverageMeter()
    order_losses = AverageMeter()

    end = time.time()
    ce_loss = nn.CrossEntropyLoss()
    mse_loss =nn.MSELoss()
    for idx, (patch1, patch2, gt_patch1, randperm) in enumerate(train_loader):
        bsz = patch1.shape[0]

        patch1 = patch1.cuda(non_blocking=True)
        patch2 = patch2.cuda(non_blocking=True)
        gt_patch1 = gt_patch1.cuda(non_blocking=True)
        randperm = randperm.long().cuda(non_blocking=True)

        pred_order1_s, pred_restore1_s, global_embd1_s = student(patch1, randperm)
        
        _, _, global_embd2_t = teacher(patch2, randperm)
        _, _, global_embd2_s = student(patch2, randperm)
        _, _, global_embd1_t = teacher(patch1, randperm)

        global_embd1_s = F.normalize(global_embd1_s, p=2.0, dim=1, eps=1e-12, out=None) # embedding1:[B,196,1024], global_embd1:[B,8192]
        global_embd2_t = F.normalize(global_embd2_t, p=2.0, dim=1, eps=1e-12, out=None)
        global_embd2_s = F.normalize(global_embd2_s, p=2.0, dim=1, eps=1e-12, out=None)
        global_embd1_t = F.normalize(global_embd1_t, p=2.0, dim=1, eps=1e-12, out=None)
        # print(len(global_embd2))

        randperm = randperm.reshape(-1)
        gt_patch1 = gt_patch1.reshape(pred_restore1_s.shape)

        order_loss = ce_loss(pred_order1_s, randperm)
        restore_loss = mse_loss(pred_restore1_s, gt_patch1)

        global_loss = mse_loss(global_embd1_s, global_embd2_t)+mse_loss(global_embd2_s, global_embd1_t)
        loss = order_loss + restore_loss + global_loss*1e5

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=log_writter)
            sys.exit(1)

        # updata metric
        order_losses.update(order_loss.item(), bsz)
        restor_losses.update(restore_loss.item(), bsz)
        global_losses.update(global_loss.item(), bsz)
        losses.update(loss.item(), bsz)

        writer.add_scalar('train/train order loss', order_losses.val, epoch*len(train_loader)+idx)
        writer.add_scalar('train/train restore loss', restor_losses.val, epoch*len(train_loader)+idx)
        writer.add_scalar('train/train global loss', global_losses.val, epoch*len(train_loader)+idx)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=student.parameters(), create_graph=is_second_order)

        # EMA update for the teacher
        it = len(train_loader)*epoch+idx
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Order loss {orderloss.val} ({orderloss.avg})\t'
                  'Restore loss {restorloss.val} ({restorloss.avg})'
                  'Global loss {globalloss.val} ({globalloss.avg})\t'
                  'Total loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                lr=optimizer.param_groups[0]['lr'],
                orderloss=order_losses, restorloss=restor_losses,
                globalloss=global_losses, ttloss=losses),
                file=log_writter)
            log_writter.flush()

        
    return losses.avg




def test(test_loader, student, teacher, conf, epoch, writer, log_writter):
    """one epoch training"""
    student.eval()
    teacher.eval()

    matches = 0
    total = 0
    mse_loss =nn.MSELoss()
    restor_losses = AverageMeter()
    global_losses = AverageMeter()

    with torch.no_grad():
        for idx, (patch1, patch2, gt_patch1, randperm) in enumerate(test_loader):
            bsz = patch1.shape[0]

            patch1 = patch1.cuda(non_blocking=True)
            patch2 = patch2.cuda(non_blocking=True)
            gt_patch1 = gt_patch1.cuda(non_blocking=True)
            randperm = randperm.cuda(non_blocking=True)

            pred_order1_s, pred_restore1_s, global_embd1_s = student(patch1, randperm)
            _, _, global_embd2_t = teacher(patch2, randperm)
            _, _, global_embd2_s = student(patch2, randperm)
            _, _, global_embd1_t = teacher(patch1, randperm)

            global_embd1_s = F.normalize(global_embd1_s, p=2.0, dim=1, eps=1e-12, out=None) # embedding1:[B,196,1024], global_embd1:[B,8192]
            global_embd2_t = F.normalize(global_embd2_t, p=2.0, dim=1, eps=1e-12, out=None)
            global_embd2_s = F.normalize(global_embd2_s, p=2.0, dim=1, eps=1e-12, out=None)
            global_embd1_t = F.normalize(global_embd1_t, p=2.0, dim=1, eps=1e-12, out=None)

            global_loss = mse_loss(global_embd1_s, global_embd2_t)+mse_loss(global_embd2_s, global_embd1_t)
            global_losses.update(global_loss.item(), bsz)

            gt_patch1 = gt_patch1.reshape(pred_restore1_s.shape)
            restor_losses.update(mse_loss(pred_restore1_s, gt_patch1).item(), bsz)

            tp1 = pred_order1_s.argmax(dim=1)
            randperm = randperm.reshape(-1)

            # print("predicted order: ", tp1, file=log_writter)
            # print("gt order: ", randperm, file=log_writter)

            matches += (tp1 == randperm).sum()
            total += randperm.shape[0]
            # print("in test", randperm.shape[0])

            # if conf.debug_mode:
            #     break

    #matches = matches.item()
    accuracy = matches / total

    writer.add_scalar('val/test global loss', global_losses.avg, epoch)
    writer.add_scalar('val/test restor loss', restor_losses.avg,epoch)
    writer.add_scalar('val/test accuracy', accuracy,epoch)

    return accuracy, restor_losses.avg, global_losses.avg




def main(conf, log_writter):

    writer = SummaryWriter(comment='pec_popar')

    # build student and teacher model
    student, optimizer,loss_scaler,start_epoch = build_model(conf)
    teacher,_,_,_ = build_model(conf)
    for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
        param_k.data.mul_(0).add_(param_q.detach().data)
    print(student, file=log_writter)
    # there is no back propagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # build dataloader
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader_NIHchest(conf)

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(conf.TRAIN.TEACHER_M, 1,
                                               conf.TRAIN.EPOCHS, len(data_loader_train))

    # start training
    for epoch in range(start_epoch, conf.TRAIN.EPOCHS + 1):
        time1 = time.time()

        lr_ = step_decay(epoch,conf)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        if epoch==1:
            accuracy, restor_losses, global_losses = test(data_loader_val, student, teacher, conf, epoch, writer, log_writter)
            print('------validation-----')
            print('Accuracy: {}. Restoration loss: {} Global loss: {}'.format(accuracy, restor_losses, global_losses), file=log_writter)
        loss = train(data_loader_train, student, teacher, momentum_schedule, optimizer, epoch, loss_scaler, conf, writer, log_writter)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1),file = log_writter)

        # tensorboard logger
        print('loss: {}@Epoch: {}'.format(loss,epoch),file = log_writter)
        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'],epoch),file = log_writter)
        log_writter.flush()
        if epoch % 10 == 0 or epoch == 1:
            # if epoch % 30 == 0:
            #     save_file = os.path.join(conf.MODEL.OUTPUT, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #     save_model(student, teacher, optimizer, conf, epoch, save_file)

            accuracy, restor_losses, global_losses = test(data_loader_val, student, teacher, conf, epoch, writer, log_writter)
            print('Accuracy: {}. Restoration loss: {} Global loss: {}'.format(accuracy, restor_losses, global_losses), file=log_writter)
            log_writter.flush()



        # save the last model
        save_file = os.path.join(conf.MODEL.OUTPUT, 'last.pth')
        save_model(student, teacher, optimizer, conf.TRAIN.EPOCHS, save_file)

        # accuracy, restor_losses = test(data_loader_val, student, epoch, writer)
        # print('Accuracy: {}. Restoration loss: {}'.format(accuracy, restor_losses), file=conf.log_writter)
        # log_writter.flush()



if __name__ == '__main__':
    args, cfg, get_cfg = parse_option()
    get_cfg.display()
    log_writter = get_cfg.log_writter

    if cfg.TRAIN.GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

main(cfg, log_writter)