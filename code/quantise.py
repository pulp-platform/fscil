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



# --------------------------------------------------------------------------------------------------
# 1. Import library
# --------------------------------------------------------------------------------------------------

import os
import sys
MYHOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
code_loc = MYHOME + "/fscil/code"
sys.path.append(MYHOME + '/fscil/code/')
sys.path.append(MYHOME + '/fscil/dory/uart_input/')
sys.path.append(MYHOME + '/quantlab/')

import json
import types
import torch
import random
torch.cuda.empty_cache()

from lib.dataloader.FSCIL.data_utils import *
from lib.run_FSCIL import *
from copy import deepcopy
from dotmap import DotMap

# Quantlab
import quantlib.editing.graphs as qg
# import quantlib.editing.editing as qe
import quantlib.algorithms as qa
import quantlib.backends as qb
from quantlib.editing.fx.passes.pact import IntegerizePACTNetPass
from quantlib.editing.fx.util import module_of_node

from typing import Optional
import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *

from lib.run_FSCIL import metatrain_baseFSCIL, train_FSCIL
from lib.param_FSCIL import ParamFSCIL

# from __future__ import annotations
# from typing import NamedTuple, List, Union, Optional, Type

# from quantlib.editing.graphs.nn.requant import Test
# Test();


# --------------------------------------------------------------------------------------------------
# 2.1 Setup Parameters
# --------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES = "0" #use '0' if you only have singgle gpu setup
BLOCK_ARCHITECTURE = "mnetv2_x4" #"mini_resnet12"
EXPERIMENT_ID = "001"
DATA_FOLDER = code_loc + "/data"
DATASET = "cifar100"

## Load checkpoint
EVAL_LOG_DIR= code_loc + "/log/" + BLOCK_ARCHITECTURE + "/" + EXPERIMENT_ID + "/quantise"
TUNE_LOG_DIR= code_loc + "/log/" + BLOCK_ARCHITECTURE + "/" + EXPERIMENT_ID + "/finetuning"
RESUME = code_loc + "/log/" + BLOCK_ARCHITECTURE + "/" + EXPERIMENT_ID + "/meta_basetrain"

RANDOM_SEED = 8
ONNX_EXPORT_NAME = 'DORY'

## MNetv2 Parameters
WIDTH_MULT=1.0
DIM_FEATURES=256
ROUND_NEAREST=1
DROPOUT_RATE=0
NUM_WAYS=60
NUM_SHOTS=5

FINETUNE_MODE = "mix" #pretrain, metatrain, mix
EVALUATION_MODE = "MODE1" #MODE1, MODE2, MODE3, MODENC

# Optimiser parameters
METATRAIN_MAX_ITER = 10
METATRAIN_LR = 0.05*0.01
METATRAIN_LR_STEP_SIZE = 3000
METATRAIN_SGD_WEIGHT_DECAY = 0
METATRAIN_SGD_MOMENTUM = 0.9
METATRAIN_BATCH_SIZE_INFERENCE = 128
METATRAIN_BATCH_SIZE_TRAINING = 6
METATRAIN_LOSS = "MultiMarginLoss"
METATRAIN_DROPOUT = 0
METATRAIN_FROZEN = "False"
METATRAIN_LOSS_PARAM = 0.1
METATRAIN_SHARPENING_ACT = "relu"
META_REPRESENTATION="real"

PRETRAIN_MAX_ITER = 3
PRETRAIN_LR = 0.025*0.01
PRETRAIN_BATCH_SIZE = 128
PRETRAIN_LR_STEP_SIZE = 30
PRETRAIN_SGD_MOMENTUM = 0.9
PRETRAIN_REPRESENTATION = "real"
PRETRAIN_SGD_WEIGHT_DECAY = 0.0005
PRETRAIN_DROPOUT = 0.0

EVAL_BATCH_SIZE_INFERENCE = 128
EVAL_BATCH_SIZE_TRAINING = 128

PACT_DECAY=0.001


# Create LOG_DIR if doesn't exist
if not os.path.exists(EVAL_LOG_DIR):
    os.makedirs(EVAL_LOG_DIR)
if not os.path.exists(TUNE_LOG_DIR):
    os.makedirs(TUNE_LOG_DIR)

# Stats found for CIFAR 100 dataset
CIFARStats = \
    {
        'normalise':
            {
                'mean': (0.507, 0.487, 0.441),
                'std':  (0.267, 0.256, 0.276)
            },
        'quantise':
            {
                'min':   -1.9023436307907104,  # computed on the normalised images of the validation partition
                'max':   2.025362491607666,   # computed on the normalised images of the validation partition
                'scale': 0.0153426020406186578125
            },
        'D':2**19,
        'in_shape': (1,3,32,32)
    }

# --------------------------------------------------------------------------------------------------
# 2.2 Create parameters
# --------------------------------------------------------------------------------------------------
finetune_metatrain_args = DotMap()
finetune_metatrain_args.which="simulation"
finetune_metatrain_args.verbose=True
finetune_metatrain_args.logprefix=None
finetune_metatrain_args.logsuffix=None
finetune_metatrain_args.logdir=TUNE_LOG_DIR
finetune_metatrain_args.parameter=[
    ["max_train_iter", METATRAIN_MAX_ITER],
    ["data_folder", "data"],
    ["resume", RESUME],
    ["trainstage", "metatrain_baseFSCIL"],
    ["dataset", DATASET],
    ["average_support_vector_inference", "True"],
    ["random_seed", RANDOM_SEED],
    ["batch_size_training", METATRAIN_BATCH_SIZE_TRAINING],
    ["batch_size_inference", METATRAIN_BATCH_SIZE_INFERENCE],
    ["optimizer", "SGD"],
    ["sharpening_activation", METATRAIN_SHARPENING_ACT],
    ["SGDnesterov", "True"],
    ["lr_step_size", METATRAIN_LR_STEP_SIZE],
    ["representation", META_REPRESENTATION],
    ["dim_features", DIM_FEATURES],
    ["num_ways", NUM_WAYS],
    ["num_shots", NUM_SHOTS],
    ["block_architecture", BLOCK_ARCHITECTURE],
    ["learning_rate", METATRAIN_LR],
    ["width_mult", 1],
    ["round_nearest", 1],
    ["SGDweight_decay", METATRAIN_SGD_WEIGHT_DECAY],
    ["dropout", "True"],
    ["dropout_rate", METATRAIN_DROPOUT],
    ["metatrain_frozen", METATRAIN_FROZEN],
    ["metatrain_loss", METATRAIN_LOSS],
    ["loss_param", METATRAIN_LOSS_PARAM],
    ["scheduler_type", "CosineAnnealing"],
    ["scheduler_warmup_step", 100],
    ["scheduler_min_lr_scaler", 0.01],
    ["normalize_weightings", "False"],
]

finetune_pretrain_args = DotMap()
finetune_pretrain_args.which="simulation"
finetune_pretrain_args.verbose=True
finetune_pretrain_args.logprefix=None
finetune_pretrain_args.logsuffix=None
finetune_pretrain_args.logdir=TUNE_LOG_DIR
finetune_pretrain_args.parameter=[
    ["max_train_iter", PRETRAIN_MAX_ITER],
    ["data_folder", "data"],
    ["resume", RESUME],
    ["trainstage", "pretrain_baseFSCIL"],
    ["pretrainFC", "linear"],
    ["dataset", DATASET],
    ["random_seed", RANDOM_SEED],
    ["learning_rate", PRETRAIN_LR],
    ["batch_size", PRETRAIN_BATCH_SIZE],
    ["optimizer", "SGD"],
    ["SGDnesterov", "True"],
    ["lr_step_size", PRETRAIN_LR_STEP_SIZE],
    ["representation", PRETRAIN_REPRESENTATION],
    ["dim_features", DIM_FEATURES],
    ["block_architecture", BLOCK_ARCHITECTURE],
    ["width_mult", 1],
    ["round_nearest", 1],
    ["SGDweight_decay", PRETRAIN_SGD_WEIGHT_DECAY],
    ["SGDmomentum", PRETRAIN_SGD_MOMENTUM],
    ["dropout", "False"],
    ["dropout_rate", PRETRAIN_DROPOUT],
    ["advance_augment", "True"],
    ["lambda_ortho", 0.5],
]


eval_args = DotMap()
eval_args.which="simulation"
eval_args.verbose=True
eval_args.logprefix=None
eval_args.logsuffix=None
eval_args.logdir=EVAL_LOG_DIR
eval_args.parameter=[
    ["data_folder", "data"],
    ["resume", RESUME],
    ["dim_features", DIM_FEATURES],
    ["nudging_act_exp", 4],
    ["nudging_act", "doubleexp"],
    ["trainstage", "train_FSCIL"],
    ["dataset", DATASET],
    ["random_seed", RANDOM_SEED],
    ["batch_size_training", EVAL_BATCH_SIZE_TRAINING],
    ["batch_size_inference", EVAL_BATCH_SIZE_INFERENCE],
    ["num_query_training", 0],
    ["optimizer", "SGD" ],
    ["sharpening_activation", "relu" ],
    ["SGDnesterov", "True"],
    ["representation", "real"],
    ["retrain_act", "real"],
    ["num_ways", NUM_WAYS],
    ["num_shots", 200], 
    ["block_architecture", BLOCK_ARCHITECTURE ],
    ["save_confusion", "False"],
    ["SGDmomentum", 0],
    ["SGDweight_decay", 0.0],
    ["lr_step_size", 300],
]

if EVALUATION_MODE=="MODE1":
    PACT_EVAL_LR = 0.25
    eval_args.parameter += [
        ["nudging_iter", 0],
        ["retrain_iter", 0],
        ["bipolarize_prototypes", "False"],
        ["use_etf_prototypes", "False"],
        ["learning_rate", PACT_EVAL_LR],
    ]
elif EVALUATION_MODE=="MODE2":
    PACT_EVAL_LR = 0.005
    eval_args.parameter += [
        ["nudging_iter", 0],
        ["retrain_iter", 100], 
        ["bipolarize_prototypes", "True"],
        ["use_etf_prototypes", "False"],
        ["lr_step_size", 30],
        ["learning_rate", PACT_EVAL_LR],
    ]
elif EVALUATION_MODE=="MODE3":
    PACT_EVAL_LR = 0.25
    eval_args.parameter += [
        ["nudging_iter", 100],
        ["retrain_iter", 100], 
        ["bipolarize_prototypes", "True"],
        ["use_etf_prototypes", "False"],
        ["lr_step_size", 30],
        ["learning_rate", PACT_EVAL_LR],
    ]
    finetune_metatrain_args.parameter += [["nudging_iter", 100]]
    finetune_pretrain_args.parameter  += [["nudging_iter", 100]]
elif EVALUATION_MODE=="MODENC":
    PACT_EVAL_LR = 0.25
    eval_args.parameter += [
        ["nudging_iter", 0],
        ["retrain_iter", 100],
        ["bipolarize_prototypes", "False"],
        ["use_etf_prototypes", "True"],
        ["lr_step_size", 30],
        ["learning_rate", PACT_EVAL_LR],
    ]

param_eval = ParamFSCIL(eval_args)


# --------------------------------------------------------------------------------------------------
# 3.1 Functions for Fake to True Quant
# --------------------------------------------------------------------------------------------------
def get_input_channels(net):
    for node in net.graph.nodes:
        if node.op == 'call_module' and isinstance(module_of_node(net, node), (nn.Conv1d, nn.Conv2d)):
            conv = module_of_node(net, node)
            return conv.in_channels
        
# THIS IS WHERE THE BUSINESS HAPPENS!
def integerize_network(net, fix_channels, dory_harmonize, in_shape, eps_in, D):
    # All we need to do to integerize a fake-quantized network is to run the
    # IntegerizePACTNetPass on it! Afterwards, the ONNX graph it produces will
    # contain only integer operations. Any divisions in the integerized graph
    # will be by powers of 2 and can be implemented as bit shifts.
    in_shp = in_shape
    int_pass = IntegerizePACTNetPass(shape_in=in_shp, eps_in=eps_in, D=D, fix_channel_numbers=fix_channels)
    int_net = int_pass(net)
    if fix_channels:
        # we may have modified the # of input channels so we need to adjust the
        # input shape
        in_shp_l = list(in_shp)
        in_shp_l[1] = get_input_channels(int_net)
        in_shp = tuple(in_shp_l)
    if dory_harmonize:
        # the DORY harmonization pass:
        # - wraps and aligns averagePool nodes so
        #   they behave as they do in the PULP-NN kernel
        # - replaces quantized adders with DORYAdder modules which are exported
        #   as custom "QuantAdd" ONNX nodes
        dory_harmonize_pass = qb.dory.DORYHarmonizePass(in_shape=in_shp)
        int_net = dory_harmonize_pass(int_net)

    return int_net

# --------------------------------------------------------------------------------------------------
# 3.3 Functions for FLoat to Fake Quant
# --------------------------------------------------------------------------------------------------
# Convert to fake quantize for all children of net
# in this way you can call net.children.forward(x)
def convert_children_to_pact(net : nn.Module,
                config : dict,
                precision_spec_file : Optional[str] = None,
                finetuning_ckpt : Optional[str] = None):
    
    #inefficient way to produce a shell
    # final_net = convert_to_pact(net, config, precision_spec_file, finetuning_ckpt)
    final_net = deepcopy(net)

    for name, child in net.named_children():
        child_pact = convert_to_pact(child, config, precision_spec_file, finetuning_ckpt)
        setattr(final_net, name, child_pact)
    
    return final_net

# Convert to fake quantize for top level net
# only convert linear, Conv2d, and Relu
def convert_to_pact(net : nn.Module,
                config : dict,
                precision_spec_file : Optional[str] = None,
                finetuning_ckpt : Optional[str] = None):

    # config is expected to contain 3 keys for each layer type:
    # PACTConv2d, PACTLinear, PACTUnsignedAct
    # their values are dicts with keys that will be used as NameFilter
    # arguments containing the kwargs for each layer.
    # An additional dict is expected to be stored under the key "kwargs", which
    # is used as the default kwargs.
    # Under the key "harmonize", the configuration for the harmonization pass
    # should be stored.

    filter_conv2d = TypeFilter(nn.Conv2d)
    filter_linear = TypeFilter(nn.Linear)
    act_types = (nn.ReLU, nn.ReLU6)
    filter_acts = VariadicOrFilter(*[TypeFilter(t) for t in act_types])

    rhos = []
    conv_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    act_cfg = config["PACTUnsignedAct"]

    harmonize_cfg = config["harmonize"]

     # the precision_spec_file is (for example) dumped by a Bayesian Bits
    # training run and overrides the 'n_levels' spec from config.json
    if precision_spec_file is not None:
        print(f"Overriding precision specification from config.json with spec from <{precision_spec_file}>...")
        with open(precision_spec_file, 'r') as fh:
            prec_override_spec = json.load(fh)['layer_levels']
        # deal with nn.DataParallel wrapping

        if all(k.startswith('module.') for k in prec_override_spec.keys()):
            prec_override_spec = {k.lstrip('module.'):v for k,v in prec_override_spec.items()}
        for cfg in (conv_cfg, lin_cfg, act_cfg):
            appl_keys = [k for k in prec_override_spec.keys() if k in cfg.keys()]
            for k in appl_keys:
                cfg[k]['n_levels'] = prec_override_spec[k]


    def make_rules(cfg : dict, filt, rule : type):
        default_cfg = cfg["kwargs"] if "kwargs" in cfg.keys() else {}
        rho = rule(filt, **default_cfg)
        rules = [rho]
        return rules

    rhos += make_rules(conv_cfg, filter_conv2d, qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(lin_cfg, filter_linear, qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(act_cfg, filter_acts, qlr.pact.ReplaceActPACTRule)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    # now harmonize the graph according to the configuration
    harmonize_pass = HarmonizePACTNetPass(**harmonize_cfg)
    final_net = harmonize_pass(net)


    if finetuning_ckpt is not None:
        print(f"Loading finetuning ckpt from <{finetuning_ckpt}>...")
        state_dict = torch.load(finetuning_ckpt)['net']
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.lstrip('module.'):v for k,v in state_dict.items()}
        lwe._graph.net.load_state_dict(state_dict, strict=False)

    return final_net

# Generate pact controller for all children nn.module 
def get_children_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}):
    all_controllers = []
    
    for name, child in net.named_children():
        ctrl = get_pact_controllers(child, schedules, kwargs_linear, kwargs_activation)
        all_controllers += ctrl
    
    return all_controllers
    
# Generate pact controller for top level module
def get_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}):
    filter_intadd = TypeFilter(PACTIntegerAdd)
    net_nodes_intadds_dissolved = LightweightGraph.build_nodes_list(net)
    net_nodes_intadds_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd,))
    lin_modules = PACTLinearController.get_modules(net)
    act_modules = PACTActController.get_modules(net)
    intadd_modules = PACTIntegerModulesController.get_modules(net)

    lin_ctrl = PACTLinearController(lin_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)
    intadd_ctrl = PACTIntegerModulesController(intadd_modules)

    return [lin_ctrl, act_ctrl, intadd_ctrl]

def integerise(x, bit_len=8):
    # val_max = 2**bit_len-1
    int_max = (2**(bit_len-1))-1
    int_min = -(2**(bit_len-1))
    return torch.clip((x /CIFARStats['quantise']['scale']).floor(), int_min, int_max) # Integerise

def integerise_fake(x, bit_len=8):
    # val_max = 2**bit_len-1
    int_max = (2**(bit_len-1))-1
    int_min = -(2**(bit_len-1))
    return torch.clip((x/CIFARStats['quantise']['scale']).floor(), int_min, int_max) * CIFARStats['quantise']['scale']
    

# --------------------------------------------------------------------------------------------------
# 3.4 Additional function
# --------------------------------------------------------------------------------------------------
def add_missing_methods(net : nn.Module):
    def hook_fn(module, input, output):
        net.embedd_val = torch.flatten(torch.floor(output), 1)

    def forward_conv(self, x: torch.Tensor):
        hook_loc = self.conv_embedding._modules['19']._modules['_QL_REPLACED_REMOVE_REDUNDANT_POOLING_0']
        hook = hook_loc.register_forward_hook(self.hook_fn) 
        self.forward(x)
        hook.remove()
        return self.embedd_val
    
    def forward_fc(self, x:torch.Tensor):
        # x = self._modules['0'](x) #Dropout layer
        x = self._modules['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0'](x)
        return x
    
    net.hook_fn = hook_fn
    net.forward_conv = types.MethodType(forward_conv, net)
    net.fc.forward = types.MethodType(forward_fc, net.fc)



# --------------------------------------------------------------------------------------------------
# 4. Main process
# --------------------------------------------------------------------------------------------------

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------------------
# 4.1 Fetch Pretrain Model
# --------------------------------------------------------------------------------------------------
args = DotMap(**(ParamFSCIL(finetune_pretrain_args).parameters)) #either pretrain or metatrain it is just the same
args = set_up_datasets(args)
args.gpu = 0
args.log_dir = ""

# Loading model and optimizer from checkpoint
model = KeyValueNetwork(args)
optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.learning_rate,nesterov=args.SGDnesterov, 
                        weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)
scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)

model,_,_, _,_ = load_checkpoint(model,optimizer,scheduler,args)

mnetv2 = model.embedding



# --------------------------------------------------------------------------------------------------
# 4.2 Convert to Fake quantization
# --------------------------------------------------------------------------------------------------
# Setup GPU and log directory
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES
device = torch.device(torch.cuda.current_device()) # Current gpu
with open(code_loc+"/config_" + BLOCK_ARCHITECTURE + ".json") as json_file:
    config = json.load(json_file)

# Fake quantize
mnetv2.eval()
mnetv2_uninit_fq = convert_children_to_pact(mnetv2, config["quantize"])
controllers = get_children_pact_controllers(mnetv2_uninit_fq, **config["controller"])
model.embedding = mnetv2_uninit_fq



# --------------------------------------------------------------------------------------------------
# 4.3 Finetune fake quantized Network
# --------------------------------------------------------------------------------------------------
if (FINETUNE_MODE == "pretrain") or (FINETUNE_MODE == "mix"):
    param_finetune = ParamFSCIL(finetune_pretrain_args)
    PACT_FINETUNE_LR = PRETRAIN_LR
    PACT_SGD_MOMENTUM = PRETRAIN_SGD_MOMENTUM
    PACT_SGD_WEIGHT_DECAY = PRETRAIN_SGD_WEIGHT_DECAY

    optimizer = qa.pact.pact_optimizers.PACTSGD(model.embedding, pact_decay=PACT_DECAY, lr=PACT_FINETUNE_LR, 
                                                momentum=PACT_SGD_MOMENTUM, weight_decay=PACT_SGD_WEIGHT_DECAY)
    
    # Finetune fake quantized network andthen evaluate
    param_finetune.parameters["model"] = model
    param_finetune.parameters["load_checkpoint"] = False
    param_finetune.parameters["optim"] = optimizer
    param_finetune.parameters["scheduler"] = scheduler
    param_finetune.parameters["quant_controller"] = controllers

    model.mode = "pretrain"
    pretrain_baseFSCIL(verbose=False, **(param_finetune.parameters))

if (FINETUNE_MODE == "metatrain") or (FINETUNE_MODE == "mix"):
    param_finetune = ParamFSCIL(finetune_metatrain_args)
    PACT_FINETUNE_LR = METATRAIN_LR
    PACT_SGD_MOMENTUM = METATRAIN_SGD_MOMENTUM
    PACT_SGD_WEIGHT_DECAY = METATRAIN_SGD_WEIGHT_DECAY

    optimizer = qa.pact.pact_optimizers.PACTSGD(model.embedding, pact_decay=PACT_DECAY, lr=PACT_FINETUNE_LR, 
                                                momentum=PACT_SGD_MOMENTUM, weight_decay=PACT_SGD_WEIGHT_DECAY)
    # Finetune fake quantized network andthen evaluate
    param_finetune.parameters["model"] = model
    param_finetune.parameters["load_checkpoint"] = False
    param_finetune.parameters["optim"] = optimizer
    param_finetune.parameters["scheduler"] = scheduler
    param_finetune.parameters["quant_controller"] = controllers

    model.mode = "meta"
    metatrain_baseFSCIL(verbose=False, **(param_finetune.parameters))

# optimizer for testing/evaluation
args.resume = TUNE_LOG_DIR
model,_,_, _,_ = load_checkpoint(model,optimizer,scheduler,args)
optimizer = qa.pact.pact_optimizers.PACTSGD(model.embedding, pact_decay=PACT_DECAY, lr=PACT_EVAL_LR, 
                                            momentum=0, weight_decay=0)



# # --------------------------------------------------------------------------------------------------
# # 4.3 Testing fake quantized Network
# # --------------------------------------------------------------------------------------------------
# # optimizer for testing/evaluation
# args.resume = TUNE_LOG_DIR
# model,_,_, _,_ = load_checkpoint(model,optimizer,scheduler,args)


# optimizer = qa.pact.pact_optimizers.PACTSGD(model.embedding.fc, pact_decay=0, lr=PACT_EVAL_LR, 
#                                             momentum=0, weight_decay=0)

# # Manipulate all activation function eps
# for c in controllers:
#     for m in c.modules:
#         if isinstance(m, PACTUnsignedAct):
#             max_val = m.running_mean.data + 5 * torch.sqrt(m.running_var.data) # 4 standard deviation
#             if (m.clip_hi.data/2>max_val):
#                 m.clip_hi.data /= 2
# # Manipulate the last layer eps even more
# mnetv2_uninit_fq.conv_embedding._modules['18']._modules['2'].clip_hi.data /= 3


# # general parameters
# mnetv2.eval()
# model.mode = "meta"
# param_eval.parameters["model"] = model.cuda()
# param_eval.parameters["load_checkpoint"] = False
# param_eval.parameters["optim"] = optimizer
# param_eval.parameters["scheduler"] = scheduler
# param_eval.parameters["quant_type"] = "fake"
# param_eval.parameters["integerize"] = integerise_fake
# param_eval.parameters["last_layer"] = None
# param_eval.parameters["eps_proto"] = None

# # set last_layer and eps_proto is for integerized backward propagation
# # assign those to None to do float back propagation (backprop only executed in MODE 2 and MODE NC)
# # last_layer should be pact_linear
# last_layer = mnetv2_uninit_fq.fc._modules['1']
# eps_in = mnetv2_uninit_fq.conv_embedding._modules['18']._modules['2'].get_eps()
# eps_out = last_layer.get_eps_out(eps_in)
# eps_w = last_layer.get_eps_w()
# param_eval.parameters["last_layer"] = last_layer
# param_eval.parameters["eps_proto"] = [eps_in, eps_out, eps_w]

# # # Dont do evaluation if you want to export the quantized network, it will change the weight value in the last layer
# # # the name is train_FSCIL but it actually is an evaluation function
# # train_FSCIL(verbose=False, **(param_eval.parameters)) 



# --------------------------------------------------------------------------------------------------
# 4.4 Convert to True Quantize network + testing
# --------------------------------------------------------------------------------------------------
# optimizer for testing/evaluation
args.resume = TUNE_LOG_DIR
model,_,_, _,_ = load_checkpoint(model,optimizer,scheduler,args)
optimizer = qa.pact.pact_optimizers.PACTSGD(model.embedding.fc, pact_decay=0, lr=PACT_EVAL_LR, 
                                            momentum=0, weight_decay=0)

# Manipulate all activation function eps
for c in controllers:
    for m in c.modules:
        if isinstance(m, PACTUnsignedAct):
            max_val = m.running_mean.data + 5 * torch.sqrt(m.running_var.data) # 5 standard deviation
            if (m.clip_hi.data/2>max_val):
                m.clip_hi.data /= 2
# Manipulate the last layer eps even more
mnetv2_uninit_fq.conv_embedding._modules['18']._modules['2'].clip_hi.data /= 3

# Fake 2 true conversion
int_net = integerize_network(mnetv2_uninit_fq.to("cpu"), True, True, CIFARStats['in_shape'], CIFARStats['quantise']['scale'], CIFARStats['D'])
model.embedding = int_net


# # --------------------------------------------------------------------------------------------------
# # 4.5 Testing true quantized Network
# # --------------------------------------------------------------------------------------------------
# # Dont do evaluation (both true or fake quantize) if you want to export the quantized network, it will change the weight value in the last layer.
# # after the network integerized some function will be missing, we need to introduce it back
# add_missing_methods(int_net) 
# # only support MODE 1 for integerized testing 
# mnetv2.eval()
# model.mode = "meta"
# param_eval.parameters["model"] = model.cuda()
# param_eval.parameters["load_checkpoint"] = False
# param_eval.parameters["optim"] = optimizer
# param_eval.parameters["scheduler"] = scheduler
# param_eval.parameters["integerize"] = integerise
# param_eval.parameters["quant_type"] = "true"
# param_eval.parameters["last_layer"] = None
# param_eval.parameters["eps_proto"] = None

# # set last_layer and eps_proto is for integerized backward propagation
# # assign those to None to do float back propagation (backprop only executed in MODE 2 and MODE 3)
# # last_layer should be pact_linear
# last_layer = int_net.fc._modules['_QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0']
# eps_in = mnetv2_uninit_fq.conv_embedding._modules['18']._modules['2'].get_eps()
# eps_out = mnetv2_uninit_fq.fc._modules['1'].get_eps_out(eps_in)
# eps_w = mnetv2_uninit_fq.fc._modules['1'].get_eps_w()
# param_eval.parameters["last_layer"] = last_layer
# param_eval.parameters["eps_proto"] = [eps_in, eps_out, eps_w]

# # This train_FSCIL result will not be correct if you have run any train_FSCIL evaluation in  fake quantize model before executing this line
# train_FSCIL(verbose=False, **(param_eval.parameters)) 



# --------------------------------------------------------------------------------------------------
# 4.6 Export true quantized network to dory compatible files
# --------------------------------------------------------------------------------------------------
# Get 1 sample data for input.txt (checksum check)
trainset, train_loader, valid_loader = get_base_dataloader(args)
xb, _ = next(iter(valid_loader))
x = xb[0].unsqueeze(0)
xi = integerise(x)

# Export the model to dory compatible onnx
qb.dory.pact_export.export_net(int_net, name="quantise", out_dir=EVAL_LOG_DIR, eps_in=CIFARStats['quantise']['scale'], integerize=False, D=CIFARStats['D'], in_data=xi)



# # --------------------------------------------------------------------------------------------------
# # 4.7 Testing the deployed network (on-board)
# # --------------------------------------------------------------------------------------------------
# # Tesing on board training
# args.resume = TUNE_LOG_DIR
# model.embedding = mnetv2_uninit_fq
# model,_,_, _,_ = load_checkpoint(model,optimizer,scheduler,args)
# optimizer = qa.pact.pact_optimizers.PACTSGD(model.embedding, pact_decay=PACT_DECAY, lr=PACT_EVAL_LR, 
#                                             momentum=0, weight_decay=0)

# import fscil_serial
# eps_in = mnetv2_uninit_fq.conv_embedding._modules['18']._modules['2'].get_eps()
# last_layer = mnetv2_uninit_fq.fc._modules['1']
# eps_out = last_layer.get_eps_out(eps_in)
# eps_w = last_layer.get_eps_w()
# mdl = fscil_serial.model_serial(eps_in, eps_out, eps_w, dev='/dev/ttyUSB1', baud_rate=460800, max_class= 100, etf_vec=model.etf_vec)
# mdl.mode = "meta"
# param_eval.parameters["load_checkpoint"] = False
# param_eval.parameters["optim"] = optimizer
# param_eval.parameters["scheduler"] = scheduler
# param_eval.parameters["model"] = mdl
# param_eval.parameters["integerize"] = integerise
# param_eval.parameters["last_layer"] = None
# param_eval.parameters["eps_proto"] = None
# train_FSCIL(verbose=False, **(param_eval.parameters)) 

















# --------------------------------------------------------------------------------------------------
# 4.7 Redundant debug codes
# --------------------------------------------------------------------------------------------------
# #Comented debug function
# VAL = 0
# def hook_fn(module, input, output):
#     global VAL
#     VAL = output
# def forward_tap(net, hook_loc, x):
#     hook = hook_loc.register_forward_hook(hook_fn) 
#     net(x)
#     hook.remove()
#     return VAL

# # from quantlib.QTensor import QTensor
# # A = QTensor(x, eps=CIFARStats['quantise']['scale'])
# mnetv2_uninit_fq.eval()
# AF = mnetv2_uninit_fq.conv_embedding._modules['0']._modules['0'](xi * CIFARStats['quantise']['scale'])
# BF = mnetv2_uninit_fq.conv_embedding._modules['0']._modules['1'](AF)
# BF = mnetv2_uninit_fq.conv_embedding._modules['0']._modules['2'](BF)
# CF = mnetv2_uninit_fq.conv_embedding._modules['1']._modules['conv']._modules['0']._modules['0'](BF)
# DF = mnetv2_uninit_fq.conv_embedding._modules['1']._modules['conv']._modules['0']._modules['1'](CF)
# DF = mnetv2_uninit_fq.conv_embedding._modules['1']._modules['conv']._modules['0']._modules['2'](DF)
# # YF = mnetv2_uninit_fq.forward_tap(x)
# # ZF = mnetv2_uninit_fq.conv_embedding(x)

# int_net.cpu()
# AI = int_net.conv_embedding._modules['0']._modules['_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0'](xi.float())
# BI = int_net.conv_embedding._modules['0']._modules['_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0'](AI)
# CI = int_net.conv_embedding._modules['1']._modules['conv']._modules['0']._modules['_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1'](BI)
# DI = int_net.conv_embedding._modules['1']._modules['conv']._modules['0']._modules['_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1'](CI)
# # ZI = int_net.forward_conv(xi.double())

# eps_out = mnetv2_uninit_fq.conv_embedding._modules['0']._modules['2'].get_eps()
# eps_in = mnetv2_uninit_fq.conv_embedding._modules['0']._modules['0'].get_eps_w()*CIFARStats["quantise"]["scale"]
# modi = int_net.conv_embedding._modules['0']._modules['_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0']
# modf = mnetv2_uninit_fq.conv_embedding._modules['0']._modules['1']

# for m in controllers[1].modules:
#     controllers[1].reset_clip_bounds(m, m.init_clip)
# #     print(m.init_clip)

# A = []
# for i in range(100):
#     A += [mdl.check_act(i).unsqueeze(0)]
# A = torch.cat(A, dim=0)
# torch.save(A, "memory100nc.pt")


            # B = model.conv_embedding(x)
            # C = args.additional.embedding.forward_conv(x)

            # model_int = args.additional
            # feat, label = model_int.get_feat_replay()
            # model_int.reset_prototypes(args)
            # model_int.update_prototypes_feat(feat,label,nways_session)
            
            # model.update_feat_replay(x, target)
            # model.recalculate_prototypes_feat()
            # A = model.forward_class(x).view(-1).type(torch.int)
            # print((A==target).sum())
