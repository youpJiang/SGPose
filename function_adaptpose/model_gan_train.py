from __future__ import print_function, absolute_import, division

import time

import numpy as np
import torch
import torch.nn as nn
# add model for generator and discriminator
from torch.autograd import Variable
from torch.utils.data import DataLoader
from common.viz import plot_16j_2d

from common.camera import project_to_2d
from common.data_loader import PoseDataSet,PoseDataSet2
from common.viz import plot_16j
from function_adaptpose.poseaug_viz import plot_poseaug
from progress.bar import Bar
from utils.gan_utils import get_discriminator_accuracy
from utils.loss import diff_range_loss, rectifiedL2loss, n_mpjpe_byjoint
from utils.utils import AverageMeter, set_grad
from utils.data_utils import random_loader
import pytorch3d.transforms as torch3d
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os

def get_adv_loss(model_dis, data_real, data_fake, criterion, summary, writer, writer_name):
    device = torch.device("cuda")
    # Adversarial losses
    real_3d = model_dis(data_real)
    fake_3d = model_dis(data_fake)

    real_label_3d = Variable(torch.ones(real_3d.size())).to(device)
    fake_label_3d = Variable(torch.zeros(fake_3d.size())).to(device)

    # adv loss
    # adv_3d_loss = criterion(real_3d, fake_3d)    # choice either one

    adv_3d_real_loss = criterion(real_3d, fake_label_3d)
    adv_3d_fake_loss = criterion(fake_3d, real_label_3d)
    # Total discriminators losses
    # adv_3d_loss = (adv_3d_real_loss + adv_3d_fake_loss) * 0.5
    adv_3d_loss =  adv_3d_fake_loss* 0.5
    # monitor training process
    ###################################################
    real_acc = get_discriminator_accuracy(real_3d.reshape(-1), real_label_3d.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_3d.reshape(-1), fake_label_3d.reshape(-1))
    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_adv_loss'.format(writer_name), adv_3d_loss.item(),
                      summary.train_iter_num)
    return adv_3d_loss


def train_dis(model_dis, data_real, data_fake, criterion, summary, writer, writer_name, fake_data_pool, optimizer):
    device = torch.device("cuda")
    optimizer.zero_grad()

    data_real = data_real.clone().detach().to(device)
    data_fake = data_fake.clone().detach().to(device)
    # store the fake buffer for discriminator training.
    data_fake = Variable(torch.Tensor(np.asarray(fake_data_pool(np.asarray(data_fake.cpu().detach()))))).to(device)

    # predicte the label
    real_pre = model_dis(data_real)
    fake_pre = model_dis(data_fake)

    real_label = Variable(torch.ones(real_pre.size())).to(device)
    fake_label = Variable(torch.zeros(fake_pre.size())).to(device)
    dis_real_loss = criterion(real_pre, real_label)
    dis_fake_loss = criterion(fake_pre, fake_label)

    # Total discriminators losses
    dis_loss = (dis_real_loss + dis_fake_loss) * 0.5

    # record acc
    real_acc = get_discriminator_accuracy(real_pre.reshape(-1), real_label.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_pre.reshape(-1), fake_label.reshape(-1))

    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_dis_loss'.format(writer_name), dis_loss.item(), summary.train_iter_num)

    # Update generators
    ###################################################
    dis_loss.backward()
    nn.utils.clip_grad_norm_(model_dis.parameters(), max_norm=1)
    optimizer.step()
    return real_acc, fake_acc


def get_diff_loss(args, bart_rlt_dict, summary, writer):
    '''
    control the modification range
    '''
    diff_loss_dict = {}
    diff_log_dict = {}

    # regulation loss for bart to avoid gan collapse
    angle_diff = bart_rlt_dict['ba_diff']  # 'ba_diff' bx15;
    angle_diff_loss,_ = diff_range_loss(torch.mean(angle_diff, dim=-1), args.ba_range_m, args.ba_range_w)

    diff_loss_dict['loss_diff_angle'] = angle_diff_loss.mean()
    diff_log_dict['log_angle_diff'] = angle_diff.detach().mean()  # record in cos_angle
    # record each bone angle
    for i in range(bart_rlt_dict['ba_diff'].shape[1]):
        diff_log_dict['log_angle@bone_{:0>2d}'.format(i)] = \
            torch.acos(torch.clamp((1 - angle_diff.detach())[:, i], -1, 1)).mean() * 57.29  # record in angle degree

    blr = bart_rlt_dict['blr']

    blr_loss = rectifiedL2loss(blr, args.blr_limit)  # blr_limit

    diff_loss_dict['loss_diff_blr'] = blr_loss.mean()
    diff_log_dict['log_diff_blr'] = blr.detach().mean()

    for key in diff_log_dict:
        writer.add_scalar('train_G_iter_diff_log/' + key, diff_log_dict[key].item(), summary.train_iter_num)

    loss = 0
    for key in diff_loss_dict:
        loss = loss + diff_loss_dict[key]
        writer.add_scalar('train_G_iter_diff_loss/' + key, diff_loss_dict[key].item(), summary.train_iter_num)
    return loss


def get_feedback_loss(args, model_pos, criterion, summary, writer,
                      inputs_2d, inputs_3d, outputs_2d_ba, outputs_3d_ba, outputs_2d_rt, outputs_3d_rt,target_2d,cam_param,pad):
    def get_posenet_loss(input_pose_2d, target_pose_3d):
        predict_pose_3d = model_pos(input_pose_2d).view(num_poses, -1, 3) #.view(num_poses, -1)
         
        target_pose_3d_rooted = target_pose_3d[:, 0,pad,:, :] - target_pose_3d[:, 0,pad,:1, :]  # ignore the 0 joint
        posenet_loss = torch.norm(predict_pose_3d - target_pose_3d_rooted, dim=-1)  # return b x j loss
        weights = torch.Tensor([1, 5, 2, 1, 5, 2, 1, 1, 1, 1, 5, 2, 1, 5, 2, 1]).to(target_pose_3d.device).unsqueeze(0)
        posenet_loss = posenet_loss * weights

        posenet_loss = torch.cat([
            torch.mean(posenet_loss[:, [1, 4, 10, 13]], dim=-1, keepdim=True),
            torch.mean(posenet_loss[:, [2, 5, 9, 11, 14]], dim=-1, keepdim=True),
            torch.mean(posenet_loss[:, [3, 6, 12, 15]], dim=-1, keepdim=True)
        ], dim=-1)
        return posenet_loss
    def get_posenet_loss_2d(input_pose_2d):
        
        predict_pose_3d = model_pos(input_pose_2d).view(num_poses, -1, 3) #.view(num_poses, -1)
        target_pose_2d=project_to_2d(predict_pose_3d.squeeze(),cam_param)
        target_pose_2d_rooted = target_pose_2d - target_pose_2d[:,:1]  # ignore the 0 joint
        input_pose_2d_rooted = input_pose_2d[:,0,pad] - input_pose_2d[:,0,pad,:1]
        posenet_loss = n_mpjpe_byjoint(input_pose_2d_rooted , target_pose_2d_rooted)  # return b x j loss

        return  posenet_loss

    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    def fix_hardratio(target_std, taget_mean, harder, easier, gloss_factordiv, gloss_factorfeedback, tag=''):
        harder_value = harder / easier

        hard_std = torch.std(harder_value)
        hard_mean = torch.mean(harder_value)

        hard_div_loss = torch.mean((hard_std - target_std) ** 2)
        hard_mean_loss, selection = diff_range_loss(harder_value, taget_mean, target_std)

        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_std'.format(tag), hard_std.mean().item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_mean'.format(tag), hard_mean.mean().item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_sample'.format(tag), harder_value[0].mean().item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_mean_loss'.format(tag), hard_mean_loss.item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_std_loss'.format(tag), hard_div_loss.item(),
                          summary.train_iter_num)
        return hard_div_loss * gloss_factordiv + hard_mean_loss * gloss_factorfeedback, selection

    # posenet loss: to generate harder case.
    # the flow: original pose --> pose BA --> pose RT
    ###################################################
    device = torch.device("cuda")
    num_poses = inputs_2d.shape[0]

    # outputs_2d_origin -> posenet -> outputs_3d_origin
    fake_pos_pair_loss_origin = get_posenet_loss(inputs_2d, inputs_3d)
    # outputs_2d_ba -> posenet -> outputs_3d_ba
    fake_pos_pair_loss_ba = get_posenet_loss(outputs_2d_ba.unsqueeze(1), outputs_3d_ba.unsqueeze(1))
    # # outputs_2d_rt -> posenet -> outputs_3d_rt
    fake_pos_pair_loss_rt = get_posenet_loss(outputs_2d_rt.unsqueeze(1), outputs_3d_rt.unsqueeze(1))
    target_2d=target_2d.unsqueeze(1).unsqueeze(2).repeat(1,1,inputs_2d.shape[2],1,1)

    target_proj_loss=get_posenet_loss_2d(target_2d)

    # pair up posenet loss
    ##########################################
    hardratio_ba = update_hardratio(args.hardratio_ba_s, args.hardratio_ba, summary.epoch, args.epochs)
    hardratio_rt = update_hardratio(args.hardratio_rt_s, args.hardratio_rt, summary.epoch, args.epochs)

    # get feedback loss
    pos_pair_loss_baToorigin, selection_ba = fix_hardratio(args.hardratio_std_ba, hardratio_ba,
                                             fake_pos_pair_loss_ba, fake_pos_pair_loss_origin,
                                             args.gloss_factordiv_ba, args.gloss_factorfeedback_ba, tag='ba')
    pos_pair_loss_rtToorigin, selection_rt = fix_hardratio(args.hardratio_std_rt, hardratio_rt,
                                             fake_pos_pair_loss_rt, fake_pos_pair_loss_origin,
                                             args.gloss_factordiv_rt, args.gloss_factorfeedback_rt, tag='rt')


    feedback_loss = pos_pair_loss_baToorigin + pos_pair_loss_rtToorigin + target_proj_loss*0.0001

    writer.add_scalar('train_G_iter_posenet_feedback/1) pos_pair_loss_origin', fake_pos_pair_loss_origin.mean().item(),
                      summary.train_iter_num)
    writer.add_scalar('train_G_iter_posenet_feedback/2) pos_pair_loss_ba', fake_pos_pair_loss_ba.mean().item(),
                      summary.train_iter_num)
    writer.add_scalar('train_G_iter_posenet_feedback/3) pos_pair_loss_rt', fake_pos_pair_loss_rt.mean().item(),
                      summary.train_iter_num)

    return feedback_loss, selection_rt

def rotation_angles(X):
    '''
    iunput= X :N*16*3
    output= N*(alpha,beta,gamma) : N*3; euler angles for N samples
    '''
    e1=X[:,4]-X[:,0]
    e2=X[:,7]-X[:,0]
    e3=torch.cross(e1,e2,dim=-1)
    e2=torch.cross(e1,e3)
    e1/=torch.norm(e1,dim=-1,keepdim=True)
    e2/=torch.norm(e2,dim=-1,keepdim=True)
    e3/=torch.norm(e3,dim=-1,keepdim=True)
    R=torch.cat((e1.unsqueeze(-1),e2.unsqueeze(-1),e3.unsqueeze(-1)),dim=-1)
    euler_angles=torch3d.matrix_to_euler_angles(R,convention=['Z','Y','X'])
    return euler_angles
def train_gan(args, poseaug_dict, data_dict, model_pos, criterion, fake_3d_sample, fake_2d_sample, summary, writer,section):
    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # extract necessary module for training.
    model_G = poseaug_dict['model_G']
    model_d3d = poseaug_dict['model_d3d']
    model_d2d = poseaug_dict['model_d2d']
    model_d2d_temp = poseaug_dict['model_d2d_temp']

    g_optimizer = poseaug_dict['optimizer_G']
    d3d_optimizer = poseaug_dict['optimizer_d3d']
    d2d_optimizer = poseaug_dict['optimizer_d2d']
    d2d_optimizer_temp = poseaug_dict['optimizer_d2d_temp']

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_G.train()
    model_d3d.train()
    model_d2d.train()
    model_d2d_temp.train()
    model_pos.train()
    end = time.time()

    # prepare buffer list for update
    tmp_3d_pose_buffer_list = []
    tmp_2d_pose_buffer_list = []
    tmp_camparam_buffer_list = []

    bar = Bar('Train pose gan', max=len(data_dict['train_gt2d3d_loader']))
    for i, ((inputs_3d, _, _, cam_param), target_d2d, target_d3d,target_d3d2) in enumerate(
            zip(data_dict['train_gt2d3d_loader'], data_dict['target_2d_loader'], data_dict['target_3d_loader'],data_dict['target_3d_loader2'])):

        if i>(section+1)*300 or i<section*300:
            continue

        pad=(inputs_3d.shape[2]-1)//2
                
        rows=torch.sum(inputs_3d==0,dim=(-1,-2,-3,-4))<(2*pad+1)*16*2
        inputs_3d=inputs_3d[rows]
        cam_param=cam_param[rows]
        target_d2d, target_d3d=target_d2d[rows], target_d3d[rows]
        lr_now = g_optimizer.param_groups[0]['lr']

        ##################################################
        #######      Train Generator     #################
        ##################################################
        set_grad([model_d3d], False)
        set_grad([model_d2d], False)
        set_grad([model_d2d_temp], False)
        set_grad([model_G], True)
        set_grad([model_pos], False)
        g_optimizer.zero_grad()

        # Measure data loading time
        data_time.update(time.time() - end)
        # if i%5==0:
        #     inputs_3d_random=random_loader(data_dict['dataset_aligned'],pad=pad)
        # else:
        inputs_3d_random=inputs_3d

        
        inputs_3d, inputs_3d_random,target_d2d,cam_param = inputs_3d.to(device), inputs_3d_random.to(device),target_d2d.to(device),cam_param.to(device)
        inputs_2d = project_to_2d(inputs_3d, cam_param)
        if args.ours2:
            #root relative
            inputs_3d_random=inputs_3d_random-inputs_3d_random[:,:,:,:1,:]
            if args.psudo_root:
                #3 psudo_root
                target_d3d=target_d3d.to(device)
                root=target_d3d[:,0,:]
            else:
                #init new root
                fx= cam_param[:,0]
                fy= cam_param[:,1]
                cx= cam_param[:,2]
                cy= cam_param[:,3]
                dX=torch.max(inputs_3d_random[:, :, :, :, 0], dim=3)[0] - torch.min(inputs_3d_random[:, :, :, :, 0], dim=3)[0]
                dY=torch.max(inputs_3d_random[:, :, :, :, 1], dim=3)[0] - torch.min(inputs_3d_random[:, :, :, :, 1], dim=3)[0]
                #1. only calculate one root for the mid frame 
                # print(dX.shape)
                # torch.Size([1024, 1, 27])
                dX=dX[:,:,pad].squeeze(dim=1)
                # print(dX.shape)
                # torch.Size([1024])
                dY=dY[:,:,pad].squeeze(dim=1)
                dxtar=target_d2d.max(dim=1)[0][:, 0] - target_d2d.min(dim=1)[0][:, 0]
                dytar=target_d2d.max(dim=1)[0][:, 1] - target_d2d.min(dim=1)[0][:, 1]
                # print(dxtar.shape)
                # torch.Size([1024])
                
                rZ=(fx*dX+fy*dY)/(dxtar+dytar) # rZ(1,1,27,1,1)
                # print(rZ.shape)
                # torch.Size([1024])
                # print(dX.shape)
                # print((fx*dX+fy*dY).shape)
                # torch.Size([1024])
                # torch.Size([1024])
                rX=rZ*(target_d2d[:,0,0]-cx)/fx
                rY=rZ*(target_d2d[:,0,1]-cy)/fy
                root=torch.stack([rX,rY,rZ],1)
            
            
            
        # poseaug: BA BL RT
        g_rlt = model_G(inputs_3d_random,target_d2d)

        # extract the generator result
        outputs_3d_ba = g_rlt['pose_ba']
        if args.ours2:
            outputs_3d_rt = g_rlt['pose_rt']+root.unsqueeze(1).unsqueeze(2) #1.
            # outputs_3d_rt = g_rlt['pose_rt']+root.unsqueeze(2) #2.
        else:
            outputs_3d_rt = g_rlt['pose_rt']
        
        # if i%300==0:
        #     Euler_angles_synthetic=rotation_angles(outputs_3d_rt[:,pad])
        #     Euler_angles=rotation_angles(inputs_3d[:,0,pad])
        #     Euler_angles_target=rotation_angles(target_d3d2)
        #     fig,axes=plt.subplots(1,3,subplot_kw=dict(projection="3d"),figsize=(15,15))
    
        #     # Creating plot
        #     axes[0].scatter(Euler_angles[:,0].cpu().detach().numpy(),Euler_angles[:,1].cpu().detach().numpy(),Euler_angles[:,2].cpu().detach().numpy(),color='r')
        #     axes[1].scatter(Euler_angles_synthetic[:,0].detach().cpu().numpy(),Euler_angles[:,1].detach().cpu().numpy(),Euler_angles_synthetic[:,2].detach().cpu().numpy(),color='b')
        #     axes[2].scatter(Euler_angles_target[:,0].detach().cpu().numpy(),Euler_angles_target[:,1].detach().cpu().numpy(),Euler_angles_target[:,2].detach().cpu().numpy(),color='g')
         
        #     os.makedirs('{}/poseaug_viz2'.format(args.checkpoint), exist_ok=True)
        #     image_name = '{}/poseaug_viz2/epoch_{:0>4d}_iter_{:0>4d}.png'.format(args.checkpoint, summary.epoch, i)
        #     plt.savefig(image_name)
        #     plt.close('all')

        outputs_2d_ba = project_to_2d(outputs_3d_ba, cam_param)  # fake 2d data
        outputs_2d_rt = project_to_2d(outputs_3d_rt, cam_param)  # fake 2d data

        # adv loss



        adv_3d_loss = get_adv_loss(model_d3d, inputs_3d[:,0,pad], outputs_3d_ba[:,pad], criterion, summary, writer, writer_name='g3d')
        
        adv_2d_loss = get_adv_loss(model_d2d, target_d2d, outputs_2d_rt[:,pad], criterion, summary, writer, writer_name='g2d')
        # diff loss. encourage diversity.
        ###################################################
        diff_loss = get_diff_loss(args, g_rlt, summary, writer)

        # posenet loss: to generate harder case.
        ###################################################
        feedback_loss,selection = get_feedback_loss(args, model_pos, criterion, summary, writer,
                                          inputs_2d, inputs_3d, outputs_2d_ba, outputs_3d_ba, outputs_2d_rt,
                                          outputs_3d_rt,target_d2d,cam_param,pad=pad) 

        if summary.epoch > args.warmup:                   
            gen_loss = adv_2d_loss * args.gloss_factord2d + \
                       adv_3d_loss * args.gloss_factord3d + \
                       feedback_loss +\
                       diff_loss * args.gloss_factordiff
        else:
            gen_loss = adv_2d_loss * args.gloss_factord2d + \
                       adv_3d_loss * args.gloss_factord3d + \
                       diff_loss * args.gloss_factordiff
                       
        

        # Update generators
        ###################################################
        if summary.train_iter_num % args.df == 0 and summary.epoch<args.tg:
            writer.add_scalar('train_G_iter/gen_loss', gen_loss.item(), summary.train_iter_num)
            writer.add_scalar('train_G_iter/lr_now', lr_now, summary.train_iter_num)
            gen_loss.backward()
            nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1)
            g_optimizer.step()

        ##################################################
        #######      Train Discriminator     #############
        ##################################################
        if summary.train_iter_num % args.df and summary.epoch<args.tg:
            set_grad([model_d3d], True)
            set_grad([model_d2d], True)
            set_grad([model_G], False)
            set_grad([model_pos], False)

            # d3d training

            train_dis(model_d3d, target_d3d, outputs_3d_ba[:,pad], criterion, summary, writer, writer_name='d3d',
                      fake_data_pool=fake_3d_sample, optimizer=d3d_optimizer)
            # d2d training

            train_dis(model_d2d, target_d2d, outputs_2d_rt[:,pad], criterion, summary, writer, writer_name='d2d',
                      fake_data_pool=fake_2d_sample, optimizer=d2d_optimizer)

        ##############################################
        # save fake data buffer for posenet training #
        ##############################################
        # here add a check so that outputs_2d_rt that out of box will be remove.
        outputs_3d_rt_selcted=outputs_3d_rt.detach()[selection.bool()]
        outputs_2d_rt_selected=outputs_2d_rt.detach()[selection.bool()]
        cam_param_selected=cam_param.detach()[selection.bool()]
        valid_rt_idx = torch.sum(outputs_2d_rt_selected > 1, dim=(1, 2,3)) < 1
        
        tmp_3d_pose_buffer_list.append(outputs_3d_rt_selcted.detach()[valid_rt_idx].cpu().numpy())
        tmp_2d_pose_buffer_list.append(outputs_2d_rt_selected.detach()[valid_rt_idx].cpu().numpy())
        tmp_camparam_buffer_list.append(cam_param_selected.detach()[valid_rt_idx].cpu().numpy())

        # update writer iter num
        summary.summary_train_iter_num_update()
          

        # plot a image for visualization
        # if i % 100 == 0:
        #     # plot_poseaug(inputs_3d[:,0,pad], inputs_2d[:,0,pad], g_rlt, cam_param, summary.epoch, i, args)
        #     plot_16j(inputs_3d[i:i+1,0,:].cpu().detach().numpy())
        #     plot_16j(g_rlt['pose_rt'][i,:].cpu().detach().numpy())
        #     plot_16j_2d(outputs_2d_rt[i].cpu().detach().numpy())
            # plot_16j(inputs_3d[100:101,0,pad].cpu().detach().numpy())
            # plot_16j(g_rlt['pose_rt'][100].cpu().detach().numpy())

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=len(data_dict['train_gt2d3d_loader']), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    bar.finish()
    ###################################
    # re-define the buffer dataloader #
    ###################################
    # buffer loader will be used to save fake pose pair
    print('\nprepare buffer loader for train on fake pose')

    if torch.sum(valid_rt_idx)>0:
        tmp_3d_pose_buffer_list=np.concatenate(tmp_3d_pose_buffer_list)
        tmp_2d_pose_buffer_list=np.concatenate(tmp_2d_pose_buffer_list)
        tmp_camparam_buffer_list=np.concatenate(tmp_camparam_buffer_list)
        print(tmp_3d_pose_buffer_list.shape)
        train_fake2d3d_loader = DataLoader(PoseDataSet2(tmp_3d_pose_buffer_list, tmp_2d_pose_buffer_list,
                                                    [['none'] * len(tmp_camparam_buffer_list)],
                                                    tmp_camparam_buffer_list,pad=0),
                                        batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers, pin_memory=True)

        data_dict['train_fake2d3d_loader'] = train_fake2d3d_loader

    return
