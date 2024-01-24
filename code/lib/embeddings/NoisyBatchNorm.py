# Copyright (C) 2022-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Authors: 
# Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
# Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
# Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)


from torch import nn
import torch

class NoisyBatchNorm2d(nn.BatchNorm2d):
  def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, noise_level=0.05):
    super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
    self.noise_level = noise_level

  def forward(self, input):
    output = super().forward(input)
    with torch.no_grad():
      noise = torch.randn(output.shape, device=output.device)
      v = (self.running_var[None,:,None,None].to(output.device))*self.noise_level
      noise = v*noise
    output = output + noise
    return output