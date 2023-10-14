from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchgeometry as tgm
import pytorch3d.transforms as torch3d


from utils.gan_utils import get_bone_lengthbypose3d, get_bone_unit_vecbypose3d, \
    get_pose3dbyBoneVec, blaugment9to15,get_BoneVecbypose3d


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(inplace=True)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)

        return y


######################################################
###################  START  ##########################
######################################################
class PoseGenerator(nn.Module):
    def __init__(self, args, input_size=16 * 3):
        super(PoseGenerator, self).__init__()
        if args.attention:
            self.BAprocess = BAGenerator_attention(input_size=16 * 3,noise_channle=45)
        else:
            self.BAprocess = BAGenerator(input_size=16 * 3,noise_channle=45)
        self.BLprocess = BLGenerator(input_size=16 * 3, blr_tanhlimit=args.blr_tanhlimit)
        if args.ours2:
            self.RTprocess = RTGenerator_ours2(input_size=16 * 3) #target
        else:
            self.RTprocess = RTGenerator(input_size=16 * 3) #target

    def forward(self, inputs_3d, target_2d):
        '''
        input: 3D pose
        :param inputs_3d: nx1x27x16x3, with hip root
        :return: nx16x3
        '''
        # print("PoseGenerator input:", inputs_3d.shape)
        # PoseGenerator input: torch.Size([1024, 1, 27, 16, 3])
        pose_ba, ba_diff = self.BAprocess(inputs_3d)  # diff may be used for div loss
        pose_bl, blr = self.BLprocess(inputs_3d, pose_ba)  # blr used for debug
        pose_rt, rt = self.RTprocess(inputs_3d,pose_bl)  # rt=(r,t) used for debug
        # print("PoseGenerator output:", pose_rt.shape)
        # PoseGenerator output: torch.Size([1024, 27, 16, 3])
        # exit()
        return {'pose_ba': pose_ba,
                'ba_diff': ba_diff,
                'pose_bl': pose_bl,
                'blr': blr,
                'pose_rt': pose_rt,
                'rt': rt}


######################################################
###################  END  ############################
######################################################

class BAGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=45, linear_size=256, num_stage=2, p_dropout=0.5):
        super(BAGenerator, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.input_size-3+15) #*2+(self.input_size-3)//3

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d):
        '''
        :param inputs_3d: nx16x3.
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        inputs_3d=inputs_3d[:,0]
        root_origin = inputs_3d[:, :,:1, :] * 1.0
        x = inputs_3d - inputs_3d[:,:, :1, :]  # x: root relative

        # extract length, unit bone vec
        bones_unit = get_bone_unit_vecbypose3d(x)
        bones_length = get_bone_lengthbypose3d(x)
        bones_vec=get_BoneVecbypose3d(x)
        middle_frame=int((x.shape[1]-1)/2)
        bones_vec=bones_vec[:,middle_frame].contiguous()
            
        # pre-processing
        bones_vec = bones_vec.view(bones_vec.size(0), -1)
        x_=x[:,middle_frame].contiguous()
        x_ = x_.view(x_.size(0), -1)
        noise = torch.randn(x_.shape[0], self.noise_channle, device=x.device)

        y = self.w1(torch.cat((x_, noise), dim=-1)) #torch.cat((bones_vec, noise), dim=-1)
  
        y = self.batch_norm1(y)

        y = self.relu(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = y.view(x.size(0), -1, 4)

        y_axis=y[:,:,:3]

        y_axis = y_axis/torch.linalg.norm(y_axis,dim=-1,keepdim=True)
        y_axis = y_axis.unsqueeze(1).repeat(1,bones_unit.shape[1],1,1)
        y_theta =y[:,:,3:4]
        y_theta=y_theta.unsqueeze(1).repeat(1,bones_unit.shape[1],1,1)
        y_theta=y_theta/x.shape[1]
        y_theta_t=torch.arange(x.shape[1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        y_theta_t=y_theta_t.repeat(bones_unit.shape[0],1,bones_unit.shape[2],1)
        y_theta=y_theta*y_theta_t

        y_axis = y_axis*y_theta
        y_rM = torch3d.axis_angle_to_matrix(y_axis.view(-1,3))[..., :3, :3]  # Nx4x4->Nx3x3 rotation matrix
        y_rM=y_rM.view(bones_unit.shape[0],bones_unit.shape[1],bones_unit.shape[2],3,3)
        modifyed_unit=torch.matmul(y_rM,bones_unit.unsqueeze(-1))[...,0]
        # # modify the bone angle with length unchanged.
        # y=y.unsqueeze(1).repeat(1,bones_unit.shape[1],1,1)
        # y=y/x.shape[1]
        # y_t=torch.arange(x.shape[1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        # y_t=y_t.repeat(bones_unit.shape[0],1,bones_unit.shape[2],bones_unit.shape[3])
        # y=y*y_t
    
        # # # print(bones_unit.shape)
        # modifyed =  bones_unit[:,middle_frame:middle_frame+1].repeat(1,y.shape[1],1,1) +y

        # modifyed_unit = modifyed / (torch.norm(modifyed, dim=-1, keepdim=True)+0.00001)

        # fix bone segment from pelvis to thorax to avoid pure rotation of whole body without ba changes.
        tmp_mask = torch.ones_like(bones_unit)
        tmp_mask[:,:, [6, 7], :] = 0.
        modifyed_unit = modifyed_unit * tmp_mask + bones_unit * (1 - tmp_mask)

        cos_angle = torch.sum(modifyed_unit * bones_unit, dim=-1)
        ba_diff = 1 - cos_angle

        modifyed_bone = modifyed_unit * bones_length

        # convert bone vec back to 3D pose
        out = get_pose3dbyBoneVec(modifyed_bone) + root_origin

        return out, ba_diff


class RTGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=45, linear_size=256, num_stage=2, p_dropout=0.5):
        super(RTGenerator, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size -> for R
        self.w1_R = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm_R = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_R = []
        for l in range(num_stage):
            self.linear_stages_R.append(Linear(self.linear_size))
        self.linear_stages_R = nn.ModuleList(self.linear_stages_R)

        # process input to linear size -> for T
        self.w1_T = nn.Linear(self.input_size + self.noise_channle, self.linear_size) 
        self.batch_norm_T = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_T = []
        for l in range(num_stage):
            self.linear_stages_T.append(Linear(self.linear_size))
        self.linear_stages_T = nn.ModuleList(self.linear_stages_T)

        # post processing

        self.w2_R = nn.Linear(self.linear_size, 7)
        self.w2_T = nn.Linear(self.linear_size, 3) 

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,inputs_3d,augx):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        inputs_3d=inputs_3d[:,0]
        middle_frame=int((inputs_3d.shape[1]-1)/2)
        pad=inputs_3d.shape[1]
        inputs_3d=inputs_3d[:,middle_frame]
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]  # x: root relative

        # pre-processing

        # x2d=target_2d[:,0,middle_frame]
        # x = torch.cat((x2d.view(x2d.size(0), -1),x3d.view(x3d.size(0), -1)),dim=-1)
        x = x.view(x.size(0), -1)

        # caculate R
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        r = self.w1_R(torch.cat((x, noise), dim=-1)) #torch.cat((x, noise), dim=1)
        r = self.batch_norm_R(r)
        r = self.relu(r)
        for i in range(self.num_stage):
            r = self.linear_stages_R[i](r)

        # r = self.w2_R(r)
        r_mean=r[:,:3]
        r_std=r[:,3:6]*r[:,3:6]
        r_axis = torch.normal(mean=r_mean,std=r_std)
        r_axis = r_axis/torch.linalg.norm(r_axis,dim=-1,keepdim=True)
        r_axis = r_axis*r[:,6:7]

        rM=torch3d.axis_angle_to_matrix(r_axis) #axis_angle
        # rM = torch3d.euler_angles_to_matrix(r_axis,["Z","Y","X"])  #euler_angle
        # rM= torch3d.quaternion_to_matrix(r_axis) #quaternion
        

        # caculate T
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        t = self.w1_T(torch.cat((x, noise), dim=-1)) #torch.cat((x, noise), dim=1)
        t = self.batch_norm_T(t)
        t = self.relu(t)
        for i in range(self.num_stage):
            t = self.linear_stages_T[i](t)

        t = self.w2_T(t)

        t[:, 2] = t[:, 2].clone() * t[:, 2].clone()
        t = t.view(x.size(0), 1, 3)  # Nx1x3 translation t

        # operat RT on original data - augx
        augx = augx - augx[:, :, :1, :]  # x: root relative
        augx = augx.permute(0, 1, 3,2).contiguous()
        rM=rM.unsqueeze(1).repeat(1,pad,1,1)
        augx_r = torch.matmul(rM, augx)
        augx_r = augx_r.permute(0,1,3, 2).contiguous()
        t=t.unsqueeze(1).repeat(1,pad,1,1)
        augx_rt = augx_r + t

        return augx_rt, (r, t)  # return r t for debug


class BLGenerator_attention(nn.Module):
    def __init__(self, input_size, noise_channle=48, linear_size=256, num_stage=2, p_dropout=0.5, blr_tanhlimit=0.2,num_heads=4,attention_size=48):
        super(BLGenerator_attention, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        # about attention 
        self.encoder = nn.Sequential(
            nn.Conv1d(3, attention_size, kernel_size=1),
            nn.BatchNorm1d(attention_size, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )  
        self.pos_embedding = nn.Parameter(torch.randn(1, 37, attention_size)) 
        self.num_heads = num_heads
        self.layer_norm = LayerNorm(attention_size)
        self.dropout = nn.Dropout(p_dropout)
        # Define multi-head attention layers
        # self.attention_layers = nn.ModuleList([MultiHeadSelfAttention(linear_size, num_heads) for _ in range(num_stage)])
        self.attention_layers =  MultiHeadSelfAttention(attention_size, num_heads)
        
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle
        self.blr_tanhlimit = blr_tanhlimit

        # 3d joints
        self.input_size = input_size + 15  # 16 * 3 + bl

        # process input to linear size -> for R
        self.w1_BL = nn.Linear( attention_size*37, self.linear_size) 
        self.batch_norm_BL = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_BL = []
        for l in range(num_stage):
            self.linear_stages_BL.append(Linear(self.linear_size))
        self.linear_stages_BL = nn.ModuleList(self.linear_stages_BL)

        # post processing
        self.w2_BL = nn.Linear(self.linear_size, 9)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d, augx):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        inputs_3d=inputs_3d[:,0]

        root_origin = inputs_3d[:, :, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :, :1, :]  # x: root relative
        # pre-processing
        x = x.view(x.size(0),x.size(1),  -1)

        # caculate blr
        bones_length_x = get_bone_lengthbypose3d(x.view(x.size(0),x.size(1),-1, 3)).squeeze(-1) 

        middle_frame=int((x.shape[1]-1)/2)
        pad=x.shape[1]
        x=x[:,middle_frame]
        bones_length_x=bones_length_x[:,middle_frame]
       
        
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        blr = torch.cat((x, bones_length_x, noise), dim=-1)
        blr = blr.reshape(x.shape[0],37,-1)
        blr = blr.permute(0, 2, 1).contiguous()# (b,37,3)->(b,3,37)
        blr = self.encoder(blr)
        blr = blr.permute(0, 2, 1).contiguous()# (b,48,37)->(b,37,48)
        blr=blr+self.pos_embedding # (b,31,48)
        # one layer attention
        # blr = blr + self.dropout(self.attention_layers(self.layer_norm(blr)))
        blr=blr.reshape(blr.shape[0],-1)
        
        blr = self.w1_BL(blr) 
        blr = self.batch_norm_BL(blr)
        blr = self.relu(blr)
        for i in range(self.num_stage):
            blr = self.linear_stages_BL[i](blr)
                        

        blr = self.w2_BL(blr)
       
        # create a mask to filter out 8th blr to avoid ambiguity (tall person at far may have same 2D with short person at close point).
        tmp_mask = torch.from_numpy(np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]]).astype('float32')).to(blr.device)
        blr = blr * tmp_mask
        # operate BL modification on original data
        blr = nn.Tanh()(blr) * self.blr_tanhlimit  # allow +-20% length change.
        blr=blr.unsqueeze(1).repeat(1,pad,1)
        bones_length = get_bone_lengthbypose3d(augx)
        augx_bl = blaugment9to15(augx, bones_length, blr.unsqueeze(3))
        return augx_bl, blr  # return blr for debug
class BLGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=48, linear_size=256, num_stage=2, p_dropout=0.5, blr_tanhlimit=0.2):
        super(BLGenerator, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle
        self.blr_tanhlimit = blr_tanhlimit

        # 3d joints
        self.input_size = input_size + 15  # 16 * 3 + bl

        # process input to linear size -> for R
        self.w1_BL = nn.Linear( self.input_size +self.noise_channle, self.linear_size) 
        self.batch_norm_BL = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_BL = []
        for l in range(num_stage):
            self.linear_stages_BL.append(Linear(self.linear_size))
        self.linear_stages_BL = nn.ModuleList(self.linear_stages_BL)

        # post processing
        self.w2_BL = nn.Linear(self.linear_size, 9)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d, augx):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        inputs_3d=inputs_3d[:,0]

        root_origin = inputs_3d[:, :, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :, :1, :]  # x: root relative
        # pre-processing
        x = x.view(x.size(0),x.size(1),  -1)

        # caculate blr
        bones_length_x = get_bone_lengthbypose3d(x.view(x.size(0),x.size(1),-1, 3)).squeeze(-1) 

        middle_frame=int((x.shape[1]-1)/2)
        pad=x.shape[1]
        x=x[:,middle_frame]
        bones_length_x=bones_length_x[:,middle_frame]
       
        
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        blr = self.w1_BL(torch.cat((x, bones_length_x, noise), dim=-1)) 
        blr = self.batch_norm_BL(blr)
        blr = self.relu(blr)
        for i in range(self.num_stage):
            blr = self.linear_stages_BL[i](blr)
                        

        blr = self.w2_BL(blr)
       
        # create a mask to filter out 8th blr to avoid ambiguity (tall person at far may have same 2D with short person at close point).
        tmp_mask = torch.from_numpy(np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]]).astype('float32')).to(blr.device)
        blr = blr * tmp_mask
        # operate BL modification on original data
        blr = nn.Tanh()(blr) * self.blr_tanhlimit  # allow +-20% length change.
        blr=blr.unsqueeze(1).repeat(1,pad,1)
        bones_length = get_bone_lengthbypose3d(augx)
        augx_bl = blaugment9to15(augx, bones_length, blr.unsqueeze(3))
        return augx_bl, blr  # return blr for debug


def random_bl_aug(x):
    '''
    :param x: nx16x3
    :return: nx16x3
    '''
    bl_15segs_templates_mdifyed = np.load('./data_extra/bone_length_npy/hm36s15678_bl_templates.npy')

    # convert 3d pose to root relative
    root = x[:, 0, :, :1, :] * 1.0
    x = x - x[:, : , : , :1, :]
    

    # extract length, unit bone vec
    bones_unit = get_bone_unit_vecbypose3d(x)

    # prepare a bone length list for augmentation.
    tmp_idx = np.random.choice(bl_15segs_templates_mdifyed.shape[0], x.shape[0])
    bones_length = torch.from_numpy(bl_15segs_templates_mdifyed[tmp_idx].astype('float32')).unsqueeze(2).unsqueeze(1)

    modifyed_bone = bones_unit * bones_length.to(x.device)

    # convert bone vec back to pose3d
    out = get_pose3dbyBoneVec(modifyed_bone)

    return out + root  # return the pose with position information.

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        batch_size, seq_length,_ = inputs.size()

        # Linear projections
        queries = self.query_linear(inputs).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        keys = self.key_linear(inputs).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        values = self.value_linear(inputs).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        # Attention calculation
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_size**0.5
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, values).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        # Output projection
        output = self.output_linear(context)
        output=output.squeeze(1)
        return output
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))
    
class BAGenerator_attention(nn.Module):
    def __init__(self, input_size, noise_channle=45, linear_size=256, num_stage=2, p_dropout=0.5, num_heads=6,attention_size=48):
        super(BAGenerator_attention, self).__init__()

        # about attention 
        self.encoder = nn.Sequential(
            nn.Conv1d(3, attention_size, kernel_size=1),
            nn.BatchNorm1d(attention_size, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )      
        self.pos_embedding = nn.Parameter(torch.randn(1, 31, attention_size)) 
        self.num_heads = num_heads
        self.layer_norm = LayerNorm(attention_size)
        self.dropout = nn.Dropout(p_dropout)
        # Define multi-head attention layers
        # self.attention_layers = nn.ModuleList([MultiHeadSelfAttention(linear_size, num_heads) for _ in range(num_stage)])
        self.attention_layers =  MultiHeadSelfAttention(attention_size, num_heads)
        
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(31*attention_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.input_size-3+15) #*2+(self.input_size-3)//3

        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, inputs_3d):
        '''
        :param inputs_3d: nx16x3.
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        inputs_3d=inputs_3d[:,0]
        root_origin = inputs_3d[:, :,:1, :] * 1.0
        x = inputs_3d - inputs_3d[:,:, :1, :]  # x: root relative

        # extract length, unit bone vec
        bones_unit = get_bone_unit_vecbypose3d(x)
        bones_length = get_bone_lengthbypose3d(x)
        bones_vec=get_BoneVecbypose3d(x)
        middle_frame=int((x.shape[1]-1)/2)
        bones_vec=bones_vec[:,middle_frame].contiguous()
            
        # pre-processing
        bones_vec = bones_vec.view(bones_vec.size(0), -1)
        x_=x[:,middle_frame].contiguous() # -> torch.Size([1024, 16, 3])
        
        x_ = x_.view(x_.size(0), -1)
        noise = torch.randn(x_.shape[0], self.noise_channle, device=x.device)
        # print(torch.cat((x_, noise).shape))
        # 16*3+45
        y=torch.cat((x_, noise), dim=-1)
        y=y.reshape(x_.shape[0],31,-1)
        y = y.permute(0, 2, 1).contiguous()# (b,31,3)->(b,3,31)
        y=self.encoder(y) # (b,3,31)->(b,24,31)
        y = y.permute(0, 2, 1).contiguous()# (b,24,31)->(b,31,24)
        # y=y+self.pos_embedding # (b,31,24)
        # one layer attention
        y = y + self.dropout(self.attention_layers(self.layer_norm(y)))
        y=y.reshape(y.shape[0],-1)
        y = self.w1(y) #torch.cat((bones_vec, noise), dim=-1)
  
        y = self.batch_norm1(y)

        y = self.relu(y)
        
        # Multi-head attention layers
        # for i in range(self.num_stage+1):
        #     y = y + self.dropout(self.attention_layers[i](self.layer_norm(y)))
        
        
        # linear layers
        # for i in range(self.num_stage-1):
        #     y = self.linear_stages[i](y)
        #     y = self.relu(y)

        y = self.w2(y)
        y = self.relu(y)
        y = y.view(x.size(0), -1, 4)

        y_axis=y[:,:,:3]

        y_axis = y_axis/torch.linalg.norm(y_axis,dim=-1,keepdim=True)
        y_axis = y_axis.unsqueeze(1).repeat(1,bones_unit.shape[1],1,1)
        y_theta =y[:,:,3:4]
        y_theta=y_theta.unsqueeze(1).repeat(1,bones_unit.shape[1],1,1)
        y_theta=y_theta/x.shape[1]
        y_theta_t=torch.arange(x.shape[1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        y_theta_t=y_theta_t.repeat(bones_unit.shape[0],1,bones_unit.shape[2],1)
        y_theta=y_theta*y_theta_t

        y_axis = y_axis*y_theta
        y_rM = torch3d.axis_angle_to_matrix(y_axis.view(-1,3))[..., :3, :3]  # Nx4x4->Nx3x3 rotation matrix
        y_rM=y_rM.view(bones_unit.shape[0],bones_unit.shape[1],bones_unit.shape[2],3,3)
        modifyed_unit=torch.matmul(y_rM,bones_unit.unsqueeze(-1))[...,0]
        # # modify the bone angle with length unchanged.
        # y=y.unsqueeze(1).repeat(1,bones_unit.shape[1],1,1)
        # y=y/x.shape[1]
        # y_t=torch.arange(x.shape[1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        # y_t=y_t.repeat(bones_unit.shape[0],1,bones_unit.shape[2],bones_unit.shape[3])
        # y=y*y_t
    
        # # # print(bones_unit.shape)
        # modifyed =  bones_unit[:,middle_frame:middle_frame+1].repeat(1,y.shape[1],1,1) +y

        # modifyed_unit = modifyed / (torch.norm(modifyed, dim=-1, keepdim=True)+0.00001)

        # fix bone segment from pelvis to thorax to avoid pure rotation of whole body without ba changes.
        tmp_mask = torch.ones_like(bones_unit)
        tmp_mask[:,:, [6, 7], :] = 0.
        modifyed_unit = modifyed_unit * tmp_mask + bones_unit * (1 - tmp_mask)

        cos_angle = torch.sum(modifyed_unit * bones_unit, dim=-1)
        ba_diff = 1 - cos_angle

        modifyed_bone = modifyed_unit * bones_length

        # convert bone vec back to 3D pose
        out = get_pose3dbyBoneVec(modifyed_bone) + root_origin

        return out, ba_diff
    
class RTGenerator_ours2(nn.Module):
    def __init__(self, input_size, noise_channle=45, linear_size=256, num_stage=2, p_dropout=0.5):
        super(RTGenerator_ours2, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size -> for R
        self.w1_R = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm_R = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_R = []
        for l in range(num_stage):
            self.linear_stages_R.append(Linear(self.linear_size))
        self.linear_stages_R = nn.ModuleList(self.linear_stages_R)

        # process input to linear size -> for T
        self.w1_T = nn.Linear(self.input_size + self.noise_channle, self.linear_size) 
        self.batch_norm_T = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_T = []
        for l in range(num_stage):
            self.linear_stages_T.append(Linear(self.linear_size))
        self.linear_stages_T = nn.ModuleList(self.linear_stages_T)

        # post processing

        self.w2_R = nn.Linear(self.linear_size, 7)
        self.w2_T = nn.Linear(self.linear_size, 3) 

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,inputs_3d,augx):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # print(inputs_3d.shape)
        # torch.Size([1024, 1, 27, 16, 3])
        
        # convert 3d pose to root relative
        inputs_3d=inputs_3d[:,0]
        middle_frame=int((inputs_3d.shape[1]-1)/2)
        pad=inputs_3d.shape[1]
        inputs_3d=inputs_3d[:,middle_frame]
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]  # x: root relative

        # pre-processing

        # x2d=target_2d[:,0,middle_frame]
        # x = torch.cat((x2d.view(x2d.size(0), -1),x3d.view(x3d.size(0), -1)),dim=-1)
        x = x.view(x.size(0), -1)

        # caculate R
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        r = self.w1_R(torch.cat((x, noise), dim=-1)) #torch.cat((x, noise), dim=1)
        r = self.batch_norm_R(r)
        r = self.relu(r)
        for i in range(self.num_stage):
            r = self.linear_stages_R[i](r)

        # r = self.w2_R(r)
        r_mean=r[:,:3]
        r_std=r[:,3:6]*r[:,3:6]
        r_axis = torch.normal(mean=r_mean,std=r_std)
        r_axis = r_axis/torch.linalg.norm(r_axis,dim=-1,keepdim=True)
        r_axis = r_axis*r[:,6:7]

        rM=torch3d.axis_angle_to_matrix(r_axis) #axis_angle
        # rM = torch3d.euler_angles_to_matrix(r_axis,["Z","Y","X"])  #euler_angle
        # rM= torch3d.quaternion_to_matrix(r_axis) #quaternion
        

        # caculate T
        # noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        # t = self.w1_T(torch.cat((x, noise), dim=-1)) #torch.cat((x, noise), dim=1)
        # t = self.batch_norm_T(t)
        # t = self.relu(t)
        # for i in range(self.num_stage):
        #     t = self.linear_stages_T[i](t)

        # t = self.w2_T(t)

        # t[:, 2] = t[:, 2].clone() * t[:, 2].clone()
        # t = t.view(x.size(0), 1, 3)  # Nx1x3 translation t

        # operat RT on original data - augx
        
        # augx = augx - augx[:, :, :1, :]  # x: root relative
        augx = augx.permute(0, 1, 3,2).contiguous()
        rM=rM.unsqueeze(1).repeat(1,pad,1,1)
        augx_r = torch.matmul(rM, augx)
        augx_r = augx_r.permute(0,1,3, 2).contiguous()
        # t=t.unsqueeze(1).repeat(1,pad,1,1)
        # augx_rt = augx_r + t

        return augx_r, (r, 0)  # return r t for debug

if __name__ == '__main__':
    # test = Project_cam3d_to_cam2d()
    random_bl_aug(None)
    print('done')
