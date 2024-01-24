# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache2.0


class Identity(object):
    """Change gt_label to one_hot encoding and keep img as the same.

    Args:
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
    """

    def __init__(self, prob=1.0):
        super(Identity, self).__init__()

        assert isinstance(prob, float) and 0.0 <= prob <= 1.0
        
        self.prob = prob

    def __call__(self, img, gt_label):
        return img, None, gt_label, None
