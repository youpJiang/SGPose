from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from progress.bar import Bar
from utils.utils import AverageMeter, set_grad


def train_posenet(bsize,model_pos, data_loader_1,data_loader_2, optimizer, criterion, device,flag):
    #此函数为了加入args.mixtrain的控制，增加了一个data_loader的输入
    # if flag表示真假数据混合: data_loader_1必须是gt2d3d，data_loader_2必须是fake
    # else表示只训练假数据或者是只训练h36m
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()

    if flag:
        # 计算每个数据集的样本数量
        num_samples_1 = len(data_loader_1.dataset)
        num_samples_2 = len(data_loader_2.dataset)
        print("dataloder1 len: " , num_samples_1)
        print("dataloder1 len: " , num_samples_2)
        num_samples= num_samples_2*2 # 按照fake的数量确定从h36m数据集中抽取的数据量

        # 计算每个数据集的权重
        weight_1 = 1 / num_samples_1
        weight_2 = 1 / num_samples_2
        # 计算每个数据集应该贡献多少样本
        num_samples_1 = num_samples_2 = num_samples // 2

        # 创建一个采用加权随机采样的采样器
        # 特别注意！！两个[]list 之间的'+'指的是concat，前后是有区别的，之前没注意这一点导致mixtrain代码大概率采样h36m的数据而不是新数据！
        sampler = data_utils.WeightedRandomSampler([weight_2] * num_samples_2 + [weight_1] * num_samples_1, num_samples_2+num_samples_1)

        # 创建一个组合的数据集，并使用采样器对其进行迭代
        # 同样此处的data_loader_2放前面或者后面是不一样的
        combined_dataset = data_utils.ConcatDataset([data_loader_2.dataset,data_loader_1.dataset])
        data_loader = data_utils.DataLoader(combined_dataset, batch_size=bsize, sampler=sampler)
    else:
        data_loader=data_loader_2
    bar = Bar('Train posenet', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        # here avoid bn with one sample in last batch, skip if num_poses=1
        if num_poses == 1:
            break

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        if len(targets_3d.shape)>3:
            pad=(targets_3d.shape[2]-1)//2
            # targets_3d=targets_3d.squeeze()
            targets_3d=targets_3d[:,0,pad]
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return