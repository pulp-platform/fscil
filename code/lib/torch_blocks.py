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
import torch as t
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------------------
def softstep(x):
    return (t.tanh(5 * (x - 1)) + 1) / 2 + (t.tanh(5 * (x + 1)) - 1) / 2

def step(x):
    return (t.sign((x - 1)) + 1) / 2 + (t.sign((x + 1)) - 1) / 2

def softabs(x, steepness=10):
    return t.sigmoid(steepness * (x - 0.5)) + t.sigmoid(steepness * (-x - 0.5))

def scaledexp(x, s=1.0): 
    return t.exp(x*s)

def softrelu(x, steepness=10):
    return t.sigmoid(steepness * (x - 0.5))

class Tanh10x(t.nn.Module):
    def __init__(self): 
        super(Tanh10x,self).__init__()

    def forward(self, x): 
        y = t.tanh(10*x)
        return y

SIM_ACT = {"bipolar": t.sign, "tanh": nn.Tanh(),"tanh10x":Tanh10x(), "real": nn.Identity(), "relu": nn.ReLU()}
# --------------------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------------------

def cosine_similarity_multi(a, b, rep = "real"):
    """
    Compute the cosine similarity between two vectors

    Parameters:
    ----------
    a:  Tensor(N_a,D)
    b:  Tensor(N_b,D)
    rep: str
        Representation to compute cosine similarity: real | bipolar | tanh
    Return 
    ------
    similarity: Tensor(N_a,N_b)
    """
    sim_act = SIM_ACT[rep]
    a_normalized = F.normalize(sim_act(a), dim=1) 
    b_normalized = F.normalize(sim_act(b), dim=1)
    similiarity = F.linear(a_normalized, b_normalized) 

    return similiarity


# --------------------------------------------------------------------------------------------------
# Layer modules
# --------------------------------------------------------------------------------------------------

class fixCos(nn.Module):
    def __init__(self, num_features, num_classes, s=1.0):
        '''
        Fixed scale alpha (given as s)
        '''
        super(fixCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = nn.Parameter(t.Tensor([s]))
        self.W = nn.Parameter(t.zeros((num_classes,num_features)))

    def forward(self, input):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # scaled dot product
        logits = self.s*F.linear(x, W)
        return logits

class myCosineLoss(nn.Module): 
    def __init__(self, rep="real"):
        super(myCosineLoss, self).__init__()
        self.sim_act = SIM_ACT[rep]
        self.cos = nn.CosineSimilarity()

    def forward(self,a,b):
        sim = self.cos(self.sim_act(a), self.sim_act(b))
        return -t.mean(sim)
    

class CustomNLLLoss(nn.Module):
    def __init__(self, correct_weight = 1):
        super(CustomNLLLoss, self).__init__()
        self.correct_weight = correct_weight
 
    def forward(self, inputs, targets): 
        # Regular NLL
        out = t.log(inputs[range(len(targets)), targets])
        return -out.sum()/len(out)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (t.device('cuda')
                  if features.is_cuda
                  else t.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = t.eye(batch_size, dtype=t.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = t.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = t.cat(t.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = t.div(
            t.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = t.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = t.scatter(
            t.ones_like(mask),
            1,
            t.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = t.exp(logits) * logits_mask
        log_prob = logits - t.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MSELoss(nn.Module):
    def __init__(self, multiplier):
        super(MSELoss, self).__init__()
        self.mul = multiplier

    def forward(self, inputs, targets): 
        with t.no_grad():
            label = -t.ones_like(inputs)
            label[:,targets] = 1.0
            label  = label * self.mul
        ret = (inputs[:,targets]-label)
        return (ret*ret).mean()


def GroupCELoss(x, group_gt, eps=1e-10):
    x_exp = t.exp(x)
    denom = t.sum(x_exp, dim=1)
    nom = t.sum(x_exp[:,group_gt], dim=1)
    loss = -t.log((nom+eps)/(denom+eps)).mean()

    return loss

def NegativeCELoss(x, group_gt, eps=1e-10):
    with t.no_grad():
        gt_label = t.argmax(x[:,group_gt], dim=1)
        gt_label = group_gt[gt_label]
    loss = F.cross_entropy(x, gt_label)

    return loss







