import os
import datetime

import torch
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import  DataLoader
import torch.optim as optim
import random
from tqdm import tqdm
from dataset.GF_loader import LevirCS_MSS_CloudSnowDataset
from model.CAN import CA_Net
from utils.components import *
from dataset import dataset_trainsformer
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)




def fast_hist(a, b, n):
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #   k形状为（HxW）的以为数组，里面的值为True 和 False,筛选符合类别的像素点
    k = (a >= 0) & (a < n)
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #   例如生成[0,0,0,0,1,3,3,2,1,0,0,0]  0代表真实是0预测也是0，1是真实是0预测是1....这些根据下面的加法都是唯一的
    #   然后用np.bincount就可以得到混肴矩阵了
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)

#   np.diag    array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
#   array是一个二维矩阵时，结果输出矩阵的对角线元素
#   np.maximum 逐元素比较大小然后选取输出
#   hist.sum(1   矩阵横着求和 ) hist.sum(0   矩阵竖着求和)两者的输出都是一维
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    # 召回率（灵敏度（Sensitivity） TP / TP + FN 竖着
    # return np.diag(hist) / np.maximum(hist.sum(1), 1)
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_class_Precision(hist):
    #精确率 TP / TP + FP 横着
    # return np.diag(hist) / np.maximum(hist.sum(0), 1)
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_Accuracy(hist):
    # ACC 准确率
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)



def compute_mIoU(pred,label,opt):
    num_classes = opt.num_classes
    pred = np.array(pred.cpu())
    label = np.array(label.cpu())
    #   创建一个全是0的矩阵，是一个混淆矩阵
    hist = np.zeros((num_classes, num_classes))
    #   获得测试集合mask图像的相对路径
    #   获取测试集合识别后的融合图像的相对路径
    # mask_imgs = [os.path.join(mask_dir, x.replace('x','y').replace('.png','.npy')) for x in png_name_list]
    # pred_imgs = [os.path.join(pred_dir, x) for x in png_name_list]
    #   读取每一个（图片-标签）对
    for ind in range(opt.batch_size//2):
        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label[ind].flatten()) != len(pred[ind].flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label[ind].flatten()), len(pred[ind].flatten())))
            continue

        #   对一张图片计算hist矩阵，并累加
        if np.sum(label[ind,:,:]==0)>0 and np.sum(pred[ind,:,:]==0)>0:
            hist += fast_hist(label[ind,:,:].flatten(), pred[ind,:,:].flatten(), num_classes)
        else:
            hist += np.array([[1,0],[0,1]])
        # 每计算36张就输出一下目前已计算的图片中所有类别平均的mIoU值
        # if name_classes is not None and ind > 0 and ind % 35 == 0:
        #     print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
        #         ind,
        #         len(label[ind]),
        #         100 * np.nanmean(per_class_iu(hist)),
        #         100 * np.nanmean(per_class_PA_Recall(hist)),
        #         100 * per_Accuracy(hist)
        #     )
        #     )
    #   计算所有结果图片的逐类别mIoU值

    # IoUs = Frequency_Weighted_Intersection_over_Union(hist)
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    ACC = per_Accuracy(hist)
    F1_score = 2 * np.multiply(Precision, PA_Recall) / (Precision + PA_Recall)
    #   逐类别输出一下mIoU值
    # if name_classes is not None:
    #     for ind_class in range(num_classes):
    #         print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
    #               + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
    #             round(Precision[ind_class] * 100, 2))  + '; F1-Score-' + str(round(F1_score[ind_class] * 100, 2)))

    #   在所有测试集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
    #     round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2))
    #       + ';F1-score: ' + str(round(np.nanmean(F1_score) * 100, 2)))
    # return   round(np.nanmean(IoUs[1]) * 100, 2), round(np.nanmean(PA_Recall[1]) * 100, 2), round(np.nanmean(Precision[1]) * 100, 2),round(np.nanmean(ACC)*100,2),round(np.nanmean(F1_score) * 100, 2)
    return round(np.nanmean(IoUs) * 100, 2), round(np.nanmean(PA_Recall) * 100, 2), round(np.nanmean(Precision) * 100, 2), round(np.nanmean(ACC) * 100, 2), round(np.nanmean(F1_score) * 100, 2)
    # return IoUs,PA_Recall,Precision,ACC,F1_score

def normPRED(d, threshold=0.47):
    # x = torch.ones(384,384)
    # y = torch.zeros(384,384)
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    dn = torch.where(dn >= 0.47, 1, 0)

    return dn


class  LevirCs_Train():
    def __init__(self,opt):
        self.opt = opt
    def train_model(self):
        set_seed(101)
        if self.opt.fp16:
            scaler = GradScaler()
        #显卡设置
        ngpus_per_node = torch.cuda.device_count()
        if self.opt.distributed:
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ['LOCAL_RANK'])
            rank = int(os.environ["RANK"])
            device = torch.device("cuda", local_rank)
            if local_rank == 1:
                print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
                print("Gpu Device Count : ", ngpus_per_node)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二张GPU设备
            torch.cuda.set_device(1)
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu0')
            local_rank = 1
            ngpus_per_node = 1
        #模型以及参数设置
        model = CA_Net()
        # weights_init(model)
        model = model.to(device)
        #断点训练
        start_epoch = 0
        if self.opt.is_resume_train:
            checkpoint = torch.load(self.opt.checkpoint_path,map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        model_train = model.train()
        if self.opt.sync_bn and ngpus_per_node > 1 and self.opt.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        elif self.opt.sync_bn:
            print("Sync_bn is not support in one gpu or not distributed.")
        if torch.cuda.is_available():
            if self.opt.distributed:
                model_train = model_train.to(device)
                model_train = torch.nn.parallel.DistributedDataParallel(model_train,device_ids=[local_rank],find_unused_parameters=True,broadcast_buffers=False,output_device=local_rank)
            else:
                model_train = torch.nn.DataParallel(model,device_ids=[1])
                cudnn.benchmark = True
                model_train = model_train.to(device)

        if local_rank == 1:
            print('---define optimizer---')
        #定义优化器
        optimizer = {
            'Adam': optim.Adam(model.parameters(), lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-3, weight_decay=1e-4),
            'Sgd': optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, nesterov=True)
        }[self.opt.optimizer]
        # if self.opt.is_resume_train:
        #     optimizer.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler_func = get_lr_scheduler(self.opt.lr_decay_type, self.opt.lr, self.opt.lr * 0.01, self.opt.epochs)
        # lf = lambda x: ((1 + math.cos(x * math.pi / self.opt.epochs)) / 2) * (1 - 0.01) + 1e-6  # cosine
        # lr_scheduler_func = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # lr_scheduler_func = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1,eta_min=0, last_epoch=- 1, verbose=False)
        # cs_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        # 定义数据
        train_joint_transform = dataset_trainsformer.Compose([
            dataset_trainsformer.RandomImgCrop(384),
            # dataset_trainsformer.RotateTransform(),  # random 90
        ])


        if self.opt.distributed:
            batch_size = self.opt.batch_size // ngpus_per_node
        else:
            batch_size = self.opt.batch_size
        if self.opt.train == 'train':
            Levircs_dataset_train = LevirCS_MSS_CloudSnowDataset(rootPath='./data/LevirCS/raw/train', joint_transform=train_joint_transform,model='train',shuffle=True)
            epoch_step_train = len(Levircs_dataset_train) / batch_size
            if local_rank == 1:
                print("---")
                print("train images: ", len(Levircs_dataset_train))
                print(self.opt.batch_size)
                print("---")


            Levircs_dataset_val = LevirCS_MSS_CloudSnowDataset(rootPath='./data/LevirCS/raw/train', joint_transform=train_joint_transform,model='val',shuffle=True)
            if local_rank == 1:
                print("---")
                print("val images: ", len(Levircs_dataset_val))
                print(self.opt.batch_size)
                print("---")
            epoch_step_val = len(Levircs_dataset_val) / batch_size
        if self.opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(Levircs_dataset_train,shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(Levircs_dataset_val,shuffle=False)
            shuffle = False
            Levircs_dataloader_train = DataLoader(Levircs_dataset_train, num_workers=self.opt.num_works,
                                              drop_last=True,shuffle=shuffle,batch_size=batch_size,
                                              pin_memory=True, sampler=train_sampler)
            Levircs_dataloader_val = DataLoader(Levircs_dataset_val, num_workers=self.opt.num_works,
                                              drop_last=True,shuffle=shuffle,batch_size=batch_size,
                                              pin_memory=True, sampler=val_sampler)
        else:
            Levircs_dataloader_train = DataLoader(Levircs_dataset_train, num_workers=self.opt.num_works,
                                                  drop_last=True, shuffle=True, batch_size=batch_size,
                                                  pin_memory=True)
            Levircs_dataloader_val = DataLoader(Levircs_dataset_val, num_workers=self.opt.num_works,
                                                drop_last=True, shuffle=True, batch_size=batch_size,
                                                pin_memory=True)
        #开始训练
        if local_rank == 1:
            print("---start training...")
            print("Start time:" + str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')))
        best_epoch = 0
        best_loss = 10000
        for epoch in range(start_epoch, self.opt.epochs):
            # Levircs_dataloader_train = DataLoader(Levircs_dataset_train, num_workers=self.opt.num_works,
            #                                       drop_last=True, shuffle=shuffle, batch_size=batch_size,
            #                                       pin_memory=True, sampler=train_sampler)
            # Levircs_dataloader_val = DataLoader(Levircs_dataset_val, num_workers=self.opt.num_works,
            #                                     drop_last=True, shuffle=shuffle, batch_size=batch_size,
            #                                     pin_memory=True, sampler=val_sampler)
            running_loss = 0.0
            running_tar_loss = 0.0
            running_loss_val = 0.0
            running_tar_loss_val = 0.0

            IoUs_sum1,IoUs_sum2, PA_Recall_sum1,PA_Recall_sum2, Precision_sum1,Precision_sum2,ACC_sum, F1_score_sum = 0, 0, 0, 0, 0, 0, 0, 0
            IoUs_sum_val1, IoUs_sum_val2, PA_Recall_sum_val1, PA_Recall_sum_val2,Precision_sum_val1, Precision_sum_val2, ACC_sum_val, F1_score_sum_val = 0, 0, 0, 0, 0, 0, 0, 0
            if self.opt.distributed:
                train_sampler.set_epoch(epoch)

            #学习率更新
            # lr_scheduler_func.step()
            lr = lr_scheduler_func(iters=epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            loss_total = []
            zhibiao_total = []
            model_train.train()#每次训练前都得进入train状态
            #进度条设置
            if local_rank == 1:
                pbar = tqdm(total=epoch_step_train/ngpus_per_node, desc=f'Epoch {epoch + 1}/{self.opt.epochs}', postfix=dict, mininterval=0.3)
            for i, data in enumerate(Levircs_dataloader_train):
                inputs, labels = data
                inputs, labels = data
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                count0 = torch.sum(labels == 0).item()+1
                count1 = torch.sum(labels == 1).item()+1
                loss_weight = torch.tensor([labels.numel()/count0,labels.numel()/count1]).to(device)
                optimizer.zero_grad()
                if not self.opt.fp16:
                    #前向传播，损失计算
                    d1,d2,d3,d4,d5 = model_train(inputs)
                    loss1, tar_loss = muti_loss_fusion(d1, d2, d3, d4, d5, labels,loss_weight)
                    running_loss += loss1.data.item()
                    running_tar_loss += tar_loss.data.item()
                    #后向传播
                    # with torch.autograd.detect_anomaly():
                    tar_loss.backward()
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print("nan gradient found")
                            print("name:", name)
                            print("param:", param.grad)
                            raise SystemExit
                    optimizer.step()
                else:
                    with autocast():
                        # 前向传播，损失计算
                        d1, d2, d3, d4, d5 = model_train(inputs)
                        loss1, tar_loss = muti_loss_fusion(d1, d2, d3, d4, d5, labels,loss_weight)
                        running_loss += loss1.data.item()
                        running_tar_loss += tar_loss.data.item()
                    # 后向传播
                    scaler.scale(tar_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                pred = torch.ones(batch_size, 384, 384)
                for j in range(batch_size):
                    # pred[j, :, :] = F.softmax(outputs[j, :, :, :].permute(1,2,0),dim=-1).argmax(axis=-1)
                    pred[j, :, :] = normPRED(d1[j, 0, :, :], threshold=0.47)
                IoUs, PA_Recall, Precision, ACC, F1_score = compute_mIoU(pred, labels, self.opt)
                IoUs_sum2 += IoUs
                PA_Recall_sum2 += PA_Recall
                Precision_sum2 += Precision
                ACC_sum += ACC
                F1_score_sum += F1_score
                if local_rank == 1:
                    pbar.set_postfix({'rl': (running_loss / (epoch_step_train)),
                                      'rtl': (running_tar_loss / ( epoch_step_train)),
                                      'iou': round(IoUs_sum2 / (i + 1), 2),
                                      'mpa': round(PA_Recall_sum2 / (i + 1), 2),
                                      'mpr': round(Precision_sum2 / (i + 1), 2),
                                      'acc': round(ACC_sum / (i + 1), 2),
                                      'F1': round(F1_score_sum / (i + 1), 2),
                                      'lr': get_lr(optimizer)},
                                     )
                    pbar.update(1)
            if local_rank == 1:
                pbar.close()
                pbar = tqdm(total=epoch_step_val/ngpus_per_node, desc=f'Epoch {epoch + 1}/{self.opt.epochs}', postfix=dict, mininterval=0.3)
            model_train.eval()
            for i, data in enumerate(Levircs_dataloader_val):
                if i >= epoch_step_val:
                    break
                inputs, labels = data
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                count0 = torch.sum(labels == 0).item()+1
                count1 = torch.sum(labels == 1).item()+1
                loss_weight = torch.tensor([labels.numel() / count0, labels.numel() / count1]).to(device)
                d1, d2, d3, d4, d5 = model_train(inputs)
                loss1_val, tar_loss_val = muti_loss_fusion(d1, d2, d3, d4, d5, labels,loss_weight)
                running_loss_val += loss1_val.data.item()
                running_tar_loss_val += tar_loss_val.data.item()
                pred = torch.ones(batch_size, 384, 384)
                for j in range(batch_size):
                    # pred[j, :, :] = d1[j, :, :, :].permute(1, 2, 0).argmax(axis=-1)
                    pred[j, :, :] = normPRED(d1[j, 0, :, :], threshold=0.47)
                IoUs, PA_Recall, Precision,ACC, F1_score = compute_mIoU(pred, labels, self.opt)
                IoUs_sum_val2 += IoUs
                PA_Recall_sum_val2 += PA_Recall
                Precision_sum_val2 += Precision
                ACC_sum_val += ACC
                F1_score_sum_val += F1_score
                if local_rank == 1:
                    pbar.set_postfix({'rl' : (running_loss_val/ ( epoch_step_val)),
                                      'rtl' : (running_tar_loss_val / ( epoch_step_val)),
                                      'iou': round(IoUs_sum_val2/(i+1),2),
                                      'mpa': round(PA_Recall_sum_val2/(i+1),2),
                                      'mpr': round(Precision_sum_val2/(i+1),2),
                                      'acc': round(ACC_sum_val/(i+1),2),
                                      'F1': round(F1_score_sum_val/(i+1),2),
                                      'lr' : get_lr(optimizer)},
                                     )
                    pbar.update(1)

            if local_rank == 1:
                pbar.close()
                print('Finish Validation')
                if ((epoch + 1) % self.opt.save_interval_epoch == 0):
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint_dict, self.opt.model_dir + "7c_{}_{}_epoch_{}".format(
                        self.opt.model, self.opt.dataset, epoch + 1))
                if (epoch % 1 == 0):
                    if best_loss > running_loss_val:
                        best_loss = running_loss_val
                        best_epoch = epoch
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optim_state_dict': optimizer.state_dict()
                        }
                        torch.save(checkpoint_dict,
                                   self.opt.model_dir + '7c_{}_{}_best_epoch'.format(self.opt.model, self.opt.dataset))
                        if local_rank == 1:
                            print(epoch)
            loss_total.append(str(running_loss/epoch_step_val))
            loss_total.append(' ')
            loss_total.append(str(running_tar_loss/epoch_step_val))
            loss_total.append('\n')
            zhibiao_total.append(str(round(IoUs_sum_val1/epoch_step_val,2)))
            zhibiao_total.append(' ')
            zhibiao_total.append(str(round(PA_Recall_sum_val1/epoch_step_val,2)))
            zhibiao_total.append(' ')
            zhibiao_total.append(str(round(Precision_sum_val1/epoch_step_val,2)))
            zhibiao_total.append(' ')
            zhibiao_total.append(str(round(ACC_sum_val / epoch_step_val, 2)))
            zhibiao_total.append(' ')
            zhibiao_total.append(str(round(F1_score_sum_val /epoch_step_val, 2)))
            zhibiao_total.append('\n')

            with open(os.path.join(self.opt.log_dir,'loss.txt'), "a+") as f:
                f.writelines(loss_total)
            with open(os.path.join(self.opt.log_dir,'zhibiao.txt'), "a+") as f2:
                f2.writelines(zhibiao_total)
            if self.opt.distributed:
                dist.barrier()
        f.close()
        f2.close()



# if __name__ == '__main__':
#     from config.Params import parser
#     opt = parser.parse_args()
#     net = LevirCs_Train(opt)
#     print(net)

