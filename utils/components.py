import torch
import math
import numpy as np
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type='cos', lr=1e-6, min_lr=0, total_iters=100, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):#0.0005,0.000005,100,3,0.0005,15
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 10)#3
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-7)#0.0005
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)#15
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return  func
def get_lr(optimizer):
    # for param_group in optimizer.param_groups:
    #     return param_group['lr']
    return optimizer.param_groups[0]['lr']
    # return optimizer.state_dict()['param_groups'][0]['lr']

ce_loss = nn.BCELoss(reduction='mean')
def muti_loss_fusion(d1, d2, d3, d4, d5, labels,loss_weight):
    loss1 = dl_loss(d1, labels,loss_weight)
    loss2 = dl_loss(d2, labels,loss_weight)
    loss3 = dl_loss(d3, labels,loss_weight)
    loss4 = dl_loss(d4, labels,loss_weight)
    loss5 = dl_loss(d5, labels,loss_weight)

    tar_loss = loss1 + loss2 + loss3 + loss4 + loss5
    return loss1,tar_loss

def muti_loss_fusion_boud(d0, d1, d2, d3, d4, d5, d6, d7, labels,loss_weight):
    loss0 = dl_loss(d0, labels,loss_weight)
    loss1 = dl_loss(d1, labels,loss_weight)
    loss2 = dl_loss(d2, labels,loss_weight)
    loss3 = dl_loss(d3, labels,loss_weight)
    loss4 = dl_loss(d4, labels,loss_weight)
    loss5 = dl_loss(d5, labels,loss_weight)
    loss6 = dl_loss(d6, labels,loss_weight)
    loss7 = dl_loss(d7, labels,loss_weight)

    tar_loss = loss0+loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
    return loss0,tar_loss

def dl_loss(pred, target,loss_weight):
    # ce_out = CE_Loss(pred, target,cls_weights=loss_weight)
    ce_out = ce_loss(pred,target)
    iou_out = iou_loss(pred, target)

    loss = ce_out + iou_out

    return loss



def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    k = 1e-6
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = (Iand1+k)/(Ior1+k)

        #IoU loss is (1-IoU1)
        IoU += (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


iou_loss = IOU(size_average=True)

def CE_Loss(inputs, target, cls_weights, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights)(temp_inputs, temp_target.long())
    return CE_loss

# if __name__ == '__main__':
#     pred = torch.zeros(4,3,224,224)
#     target = torch.zeros(4,224,224)
#     CE_Loss(pred,target.long(),None)
#     iou_loss(pred,target)



def get_cls_S(cls_string):
    """
    Get the classes from the integer values in the true masks (i.e. 'water' in sen2cor has integer value 3)
    """
    cls_int = []
    for c in cls_string:
        if c == 'shadow':
            cls_int.append(0)
            cls_int.append(1)
        elif c == 'water':
            cls_int.append(2)
            cls_int.append(6)
        elif c == 'snow':
            cls_int.append(3)
        elif c == 'cloud':
            cls_int.append(5)
        elif c == 'clear':
            cls_int.append(4)
    return cls_int



def bands_select_S(img_dataset,opt,width,height,channel):
    x = np.zeros((width, height, channel), dtype=np.uint16)
    for i, j in enumerate(opt.bands_list):
        if i == 0:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 1:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 2:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 3:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 4:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 5:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 6:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 7:
            raise ValueError('Band 8 (pan-chromatic band) cannot be included')
        elif i == 8:
            x[:, :, i] = img_dataset[:, :, j]
        elif i == 9:
            x[:, :, i] = img_dataset[:, :, j]
    return  x


def weights_init(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)