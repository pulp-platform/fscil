# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache2.0


import numpy as np
import torch

class BatchMixupLayer(object):
    r"""Mixup layer for a batch of data.

    Mixup is a method to reduces the memorization of corrupt labels and
    increases the robustness to adversarial examples. It's
    proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`

    This method simply linearly mix pairs of data and their labels.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            are in the note.
        num_classes (int): The number of classes.
        prob (float): The probability to execute mixup. It should be in
            range [0, 1]. Default sto 1.0.

    Note:
        The :math:`\alpha` (``alpha``) determines a random distribution
        :math:`Beta(\alpha, \alpha)`. For each batch of data, we sample
        a mixing ratio (marked as :math:`\lambda`, ``lam``) from the random
        distribution.
    """

    def __init__(self, alpha, prob):
        super(BatchMixupLayer, self).__init__() 
        
        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0
        
        self.prob = prob
        self.alpha = alpha

    def mixup(self, img, gt_label):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        return mixed_img, lam,  gt_label, gt_label[index]

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)


class BatchMultiMixupLayer(object):

    def __init__(self, n, prob):
        super(BatchMultiMixupLayer, self).__init__() 
        
        assert n > 1
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0
        
        self.prob = prob
        self.n = n
        self.prime = [  1,   2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,
                       41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,
                      101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
                      167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
                      239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
                      313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
                      397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                      467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563,
                      569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
                      643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727,
                      733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821,
                      823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907,
                      911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

    def mixup(self, img, gt_label):
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = img[index, :]
        for i in range(self.n-1):
            new_idx = (index + self.prime[i]) % batch_size
            mixed_img += img[new_idx, :]
        mixed_img /= self.n

        label = torch.ones_like(gt_label)*-1
        return mixed_img, None,  label, None

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)
