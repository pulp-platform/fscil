#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# Modified by:
# Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
# Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
# Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import numpy as np
import sys, os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dotmap import DotMap
from lib.nudging import nudge_prototypes
from .embeddings.ResNet12 import ResNet12
from .embeddings.resnet import resnet18
from .embeddings.ResNet20 import ResNet20
from .embeddings.mobilenetv3 import MobileNetV3
from .embeddings.mobilenetv2 import MobileNetV2
from .embeddings.efficientnet import EfficientNet
from lib.torch_blocks import fixCos, softstep, step, softabs, softrelu, cosine_similarity_multi, scaledexp
t.manual_seed(0) #for reproducability
import math
import pdb

# Function for neural collapse prototype
def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = t.tensor(orth_vec).float()
    assert t.allclose(t.matmul(orth_vec.T, orth_vec), t.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            t.max(t.abs(t.matmul(orth_vec.T, orth_vec) - t.eye(num_classes))))
    return orth_vec

# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class KeyValueNetwork(nn.Module):

    # ----------------------------------------------------------------------------------------------
    # Special Functions & Properties
    # ----------------------------------------------------------------------------------------------

    def __init__(self, args,mode="meta"):
        super().__init__()

        self.args = args
        self.mode = mode

        # Modules
        if args.block_architecture == "mini_resnet12":
            self.embedding = ResNet12(args)
        elif args.block_architecture == "mini_resnet18": 
            self.embedding = resnet18(num_classes=args.dim_features)
        elif args.block_architecture == "mini_resnet20": 
            self.embedding = ResNet20(num_classes=args.dim_features)
        elif args.block_architecture == "resnet18_pretrained": 
            self.embedding = resnet18(True, num_classes=args.dim_features)
        elif args.block_architecture == "mnetv3_small": 
            self.embedding = MobileNetV3(mode='small', num_classes=args.dim_features, input_size=32, 
                                        width_multiplier=args.width_mult, dropout = args.dropout_rate)
        elif args.block_architecture == "mnetv3_small_8x8": 
            self.embedding = MobileNetV3(mode='small_8x8', num_classes=args.dim_features, input_size=32, 
                                        width_multiplier=args.width_mult, dropout = args.dropout_rate)
        elif args.block_architecture == "mnetv2_x4": 
            # 8x8
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 1],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
            self.embedding = MobileNetV2(num_classes=args.dim_features, width_mult = args.width_mult, 
                                            inverted_residual_setting=inverted_residual_setting,
                                            round_nearest = args.round_nearest, dropout = args.dropout_rate)

        elif args.block_architecture == "mnetv2_x2": 
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
            self.embedding = MobileNetV2(num_classes=args.dim_features, width_mult = args.width_mult, 
                                            inverted_residual_setting=inverted_residual_setting,
                                            round_nearest = args.round_nearest, dropout = args.dropout_rate) 
        elif args.block_architecture == "mnetv2": # Original
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            self.embedding = MobileNetV2(num_classes=args.dim_features, width_mult = args.width_mult, 
                                            inverted_residual_setting=inverted_residual_setting,
                                            round_nearest = args.round_nearest, dropout = args.dropout_rate)        
        elif args.block_architecture == "efficientnet": 
            self.embedding = EfficientNet(num_classes=args.dim_features, width_coef=args.width_mult, depth_coef=1.0, 
                                            scale=1.0, dropout_ratio=args.dropout_rate, se_ratio=0.25)

        # Load pretrain FC module 
        fc_out = args.pretrainFC_vec if (args.pretrainFC_vec is not None) else args.base_class
        if args.pretrainFC == "spherical": # use cosine similarity
            self.fc_pretrain = fixCos(args.dim_features,fc_out)
        elif args.pretrainFC == "identity":
            self.fc_pretrain = nn.Identity()
        elif (args.pretrainFC == "rotatedetf"):
            self.fc_pretrain = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(args.dim_features),
                RotatedETFHead(args.dim_features,fc_out, args.base_class)
            )
        elif (args.pretrainFC == "fixedetf"):
            self.fc_pretrain = nn.Sequential(
                nn.Tanh(),
                nn.BatchNorm1d(args.dim_features),
                FixedETFHead(args.dim_features,fc_out, args.base_class)
            )
        else: 
            self.fc_pretrain = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(args.dim_features),
                nn.Linear(args.dim_features,fc_out,bias=False)
            )

        # Activations
        activation_functions = {
            'softabs'  :(lambda x: softabs(x, steepness=args.sharpening_strength)),
            'softrelu' :(lambda x: softrelu(x, steepness=args.sharpening_strength)),
            'relu'     :nn.ReLU(), 
            'abs'      :t.abs,
            'scaledexp':(lambda x: scaledexp(x, s = args.sharpening_strength)),
            'exp'      :t.exp,
            'real'     :lambda x : x + 1
        }
        approximations = {
            'softabs':  'abs',
            'softrelu': 'relu'
        }
        
        self.sharpening_activation = activation_functions[args.sharpening_activation]

        # Access to intermediate activations
        self.intermediate_results = dict()
        
        self.sum_cnt_vec = t.zeros((args.num_classes,1)).cuda(args.gpu)
        self.sum_feat_vec = t.zeros((args.num_classes,self.embedding.n_interm_feat)).cuda(args.gpu)
        self.sum_square_vec = t.zeros((args.num_classes,self.embedding.n_interm_feat)).cuda(args.gpu)

        self.feat_replay = t.zeros((args.num_classes,self.embedding.n_interm_feat)).cuda(args.gpu)
        self.label_feat_replay = t.diag(t.ones(self.args.num_classes)).cuda(args.gpu)

    # ----------------------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------------------

    def forward(self, inputs):
        '''
        Forward pass of main model

        Parameters:
        -----------
        inputs:  Tensor (B,H,W)
            Input data
        Return: 
        -------
        output:  Tensor (B,ways)
        ''' 
        # Embed batch
        # query_vectors = self.embedding(inputs) #oldway
        if hasattr(self.embedding, "forward_conv"):
            self.activation_memory = self.embedding.forward_conv(inputs)
        else:
            self.activation_memory = self.embedding.conv_embedding(inputs)
        query_vectors = self.embedding.fc(self.activation_memory)
        self.proto = query_vectors

        if self.mode =="pretrain":
            output =  self.fc_pretrain(query_vectors)

        else: #elif self.mode =="meta": 
            ##################### Cosine similarities #########################################################
            self.similarities = cosine_similarity_multi(query_vectors, self.key_mem, rep=self.args.representation)
          
            ################# Sharpen the similarities with a soft absolute activation ############################
            similarities_sharpened = self.sharpening_activation(self.similarities)
                
            # Normalize the similarities in order to turn them into weightings
            if self.args.normalize_weightings:
                denom = t.sum(similarities_sharpened, dim=1, keepdim=True)
                weightings = t.div(similarities_sharpened, denom)
            else:
                weightings = similarities_sharpened

            # Return weighted sum of labels
            if self.args.average_support_vector_inference:
                output = weightings
            else:
                output = t.matmul(weightings, self.val_mem)

        return output

    def write_mem(self,x,y):
        '''
        Rewrite key and value memory

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B,w)
            One-hot encoded classes
        ''' 
        self.key_mem = self.embedding(x)
        self.val_mem = y

        if self.args.average_support_vector_inference:
            self.key_mem = t.matmul(t.transpose(self.val_mem,0,1), self.key_mem)
        return


    def reset_prototypes(self,args): 
        if hasattr(self,'key_mem'):
            self.key_mem.data.fill_(0.0)
        else: 
            self.key_mem = nn.parameter.Parameter(t.zeros(self.args.num_classes, self.args.dim_features),requires_grad=False).cuda(args.gpu)
            self.val_mem = nn.parameter.Parameter(t.diag(t.ones(self.args.num_classes)),requires_grad=False).cuda(args.gpu)

    def update_prototypes(self,x,y): 
        '''
        Update key memory  

        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables 
        ''' 

        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        prototype_vec = t.matmul(t.transpose(y_onehot,0,1), support_vec)
        self.key_mem.data += prototype_vec

    def bipolarize_prototypes(self):
        '''
        Bipolarize key memory   
        '''
        return t.sign(self.key_mem.data)
    
    def use_etf_prototypes(self):
        '''
        Bipolarize key memory   
        '''
        return self.fc_pretrain[2].etf_vec.t()

    def get_sum_support(self,x,y):
        '''
        Compute prototypes
        
        Parameters:
        -----------
        x:  Tensor (B,D)
            Input data
        y:  Tensor (B)
            lables 
        '''
        support_vec = self.embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        sum_cnt = t.sum(y_onehot,dim=0).unsqueeze(1)
        sum_support = t.matmul(t.transpose(y_onehot,0,1), support_vec)
        return sum_support, sum_cnt


    def update_feat_replay(self,x,y): 
        '''
        Compute feature representatin of new data and update
        Parameters:
        -----------
        x   t.Tensor(B,in_shape)
            Input raw images
        y   t.Tensor (B)
            Input labels

        Return: 
        -------
        '''
        # TODO: fix the architecture to provide forward_conv
        if hasattr(self.embedding, "forward_conv"):
            feat_vec = self.embedding.forward_conv(x)
        else:
            feat_vec = self.embedding.conv_embedding(x)
        y_onehot = F.one_hot(y, num_classes = self.args.num_classes).float()
        self.sum_cnt_vec += t.sum(y_onehot,dim=0).unsqueeze(1)
        self.sum_feat_vec += t.matmul(t.transpose(y_onehot,0,1), feat_vec)
        self.sum_square_vec += t.matmul(t.transpose(y_onehot,0,1), feat_vec*feat_vec)

    def get_feat_replay(self, noise_level=0): 
        if noise_level==0:
            self.feat_replay  = t.div(self.sum_feat_vec,self.sum_cnt_vec+1e-8)
            return self.feat_replay, self.label_feat_replay
        else:
            self.feat_replay  = t.div(self.sum_feat_vec,self.sum_cnt_vec+1e-8)
            self.feat_replay_std = self.sum_square_vec - self.feat_replay*self.feat_replay
            self.feat_replay_std  = t.sqrt(t.div(self.feat_replay_std,self.sum_cnt_vec+1e-8))
            r = t.randn(self.feat_replay.shape, device=self.feat_replay.device)
            return self.feat_replay + r*noise_level*self.feat_replay_std, self.label_feat_replay

    def update_prototypes_feat(self,feat,y_onehot,nways=None): 
        '''
        Update key 

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        y:  Tensor (B)
        nways: int
            If none: update all prototypes, if int, update only nwyas prototypes
        ''' 
        support_vec = self.get_support_feat(feat)
        prototype_vec = t.matmul(t.transpose(y_onehot,0,1), support_vec)

        if nways is not None: 
            self.key_mem.data[:nways] += prototype_vec[:nways]
        else:
            self.key_mem.data += prototype_vec

    def get_support_feat(self,feat): 
        '''
        Pass activations through final FC 

        Parameters:
        -----------
        feat:  Tensor (B,d_f)
            Input data
        Return:
        ------
        support_vec:  Tensor (B,d)
            Mapped support vectors
        ''' 
        support_vec = self.embedding.fc(feat)
        return support_vec

    def nudge_prototypes(self,num_ways,writer,session,gpu): 
        '''
        Prototype nudging
        Parameters:
        -----------
        num_ways:   int
        writer:     Tensorboard writer
        session:    int
        gpu:        int

        ''' 
        prototypes_orig = self.key_mem.data[:num_ways]
        self.key_mem.data[:num_ways]  = nudge_prototypes(prototypes_orig,writer,session,
                                                        gpu=self.args.gpu,num_epochs=self.args.nudging_iter,
                                                        bipolarize_prototypes=self.args.bipolarize_prototypes,
                                                        act=self.args.nudging_act,
                                                        act_exp = self.args.nudging_act_exp)
        return

    def hrr_superposition(self,num_ways,nsup=2): 
        '''
        Compression an retrieval of EM with HRR
        Parameters: 
        ----------
        num_ways: Number of active ways, if not specified entire memory will be bipolarized
        nsup: number of superimposed vectors 
        '''
        n_comp = math.ceil(num_ways/nsup)
        for m in range(n_comp):        
            # generate a new set of keys
            key =1/math.sqrt(self.args.dim_features)*t.randn((nsup,self.args.dim_features))
            superpos = t.zeros(self.args.dim_features,1).cuda(self.args.gpu) 
            for way in range(nsup): 
                rotMat = t.FloatTensor().set_(key[way].repeat(2).storage(), storage_offset=0, 
                                                size=t.Size((self.args.dim_features,self.args.dim_features)), 
                                                stride=(1, 1)).cuda(self.args.gpu) 
                superpos = superpos + t.mm(rotMat,self.key_mem.data[way+m*nsup].view(-1,1))

            # retrieval 
            for way in range(nsup):
                if way+m*nsup<num_ways: # only restore if needed 
                    rotMat = t.FloatTensor().set_(key[way].repeat(2).storage(), storage_offset=0, 
                                                    size=t.Size((self.args.dim_features,self.args.dim_features)),
                                                    stride=(1, 1)).cuda(self.args.gpu) 
                    self.key_mem.data[way+m*nsup] = t.mm(rotMat,superpos).squeeze()



class ContrastiveHead(nn.Module):

    def __init__(self, in_feat, out_feat, mid_feat, overlap=0.0):
        super().__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.mid_feat = mid_feat
        self.overlap_feat = round(overlap*mid_feat)
        self.non_overlap_feat = mid_feat - self.overlap_feat

        self.head = t.nn.Sequential(
            t.nn.ReLU(),
            t.nn.BatchNorm1d(mid_feat),
            t.nn.Linear(mid_feat,out_feat)
        )

        if (self.non_overlap_feat != 0):
            self.extractor = t.nn.Linear(in_feat,self.non_overlap_feat)
    
    def forward(self, feat, x):
        if (self.non_overlap_feat == 0):
            return self.head(feat)
        elif (self.overlap_feat == 0):
            return self.head(self.extractor(x))
        else:
            y = self.extractor(x)
            y = t.cat([y,feat[:,:self.overlap_feat]], dim=1)
            return self.head(y)


def get_random_rotation(dim=2):
    assert dim>=2
    rot_ary = t.rand(dim)*2*np.pi - np.pi
    rcos  = t.cos(rot_ary)
    rsin  = t.sin(rot_ary)
    rmats = t.eye(dim).unsqueeze(0).repeat(dim,1,1)

    idx0 = t.arange(dim)
    idx1 = (idx0 + 1) % dim
    rmats[idx0, idx0, idx0] =  rcos # top left
    rmats[idx0, idx1, idx0] = -rsin # top right
    rmats[idx0, idx0, idx1] =  rsin # bottom left
    rmats[idx0, idx1, idx1] =  rcos # bottom right

    combined_rmat = rmats[0]
    for m in rmats[1:]:
        combined_rmat = combined_rmat @ m

    return combined_rmat

class FixedETFHead(nn.Module):
    def __init__(self, in_feat, out_feat, base_class):
        super().__init__()
        assert out_feat<=in_feat
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.base_class = base_class
        
        # Generate etf vector
        orth_vec = generate_random_orthogonal_matrix(in_feat, out_feat)
        i_nc_nc = t.eye(out_feat)
        one_nc_nc: t.Tensor = t.mul(t.ones(out_feat, out_feat), (1 / out_feat))
        etf_vec = t.mul(t.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(out_feat / (out_feat - 1)))
        etf_vec = F.normalize(etf_vec, p=2, dim=0)
        # register etf_vec
        self.register_buffer('etf_vec', etf_vec)

        # Find null of etf vector
        etf_null = etf_vec[:,:base_class].detach().cpu().numpy()
        u, s, vh = np.linalg.svd(etf_null)
        etf_null = u[:,base_class:]
        etf_null = t.tensor(etf_null)
        # register null vector
        self.register_buffer('etf_null', etf_null)

        self.etf_mag = nn.Parameter(t.ones(1, out_feat), requires_grad=True)
    
    def forward(self, x):
        etf_vec = (self.etf_vec * self.etf_mag)
        result = x @ etf_vec
        
        return result


class RotatedETFHead(nn.Module):

    def __init__(self, in_feat, out_feat, base_class):
        super().__init__()
        assert out_feat<=in_feat
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.base_class = base_class

        # I set this to false as I find out that the rotataion does not improve the result
        # I change it from false to true in run_FSCIL.py
        self.orthonormal = nn.Parameter(t.eye(in_feat), requires_grad=False)
        
        self.fixedetf = FixedETFHead(in_feat, out_feat, base_class)
    
    def get_rotation_mat(self):
        if self.training: #check if it is in module.eval() / module.train() mode
            with t.no_grad():
                # dat = self.orthonormal.data
                # dat =  F.normalize(dat, dim=0)
                # dat =  F.normalize(dat, dim=1)
                # self.orthonormal.data = dat
                # DEBUG
                dat = self.orthonormal.data
                dat, _ = t.linalg.qr(dat)
                self.orthonormal.data = dat

        return self.orthonormal
    
    @property
    def etf_vec(self):
        combined_rmat = self.get_rotation_mat()
        return combined_rmat @ self.fixedetf.etf_vec
    
    @property
    def etf_null(self):
        combined_rmat = self.get_rotation_mat()
        return combined_rmat @ self.fixedetf.etf_null
    
    def forward(self, x):
        combined_rmat = self.get_rotation_mat()
        result = self.fixedetf(x @ combined_rmat)

        return result
