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


import datetime
from copy import deepcopy
from ast import literal_eval
import torch as t
import numpy as np
import random


# --------------------------------------------------------------------------------------------------
# Parameter related functions
# --------------------------------------------------------------------------------------------------

def test_and_cast(key, value):
    integers = ['num_ways', 'num_ways_training', 'num_ways_inference', 'num_shots', 'num_shots_training',
                'num_shots_inference', 'num_query_training', 'num_query_inference','batch_size', 
                'batch_size_training', 'batch_size_inference','nudging_act_exp',
                'max_train_iter', 'max_val_iter', 'max_test_iter', 'validation_frequency', 'moving_average_samples',
                'summary_frequency_often', 'dim_features', 'summary_frequency_once', 'summary_frequency_very_often', 
                'summary_frequency_seldom', 'random_seed','retrain_iter','nudging_iter', 'lr_step_size', 'num_workers',
                'em_compression_nsup', 'round_nearest', 'pretrainFC_vec', 'ETFpoints', 'scheduler_warmup_step']
    floats = ['learning_rate', 'norm_weight', 'sharpening_strength', 'regularization', 'dropout_rate', 
               'SGDmomentum','SGDweight_decay', 'width_mult', 'loss_param', 'lambda_ortho', 'lambdaCL', 
               'temperatureCL', 'overlapCL', 'scheduler_min_lr_scaler', 'scheduler_gamma', 'scheduler_start_lr_scaler']
    strings = ['block_architecture','block_interface','sharpening_activation', 'log_dir', 'representation', 'log',
               'data_folder', 'experiment_dir', 'dataset','trainstage', 'resume','pretrainFC', 'optimizer',
               'retrain_act','nudging_act','em_compression', 'metatrain_loss', 'pretrain_loss', 'fscil_loss', 'typeCL',
               'scheduler_type']
    bools = ['with_repetition', 'allow_empty_classes', 'normalize_weightings','average_support_vector_inference', 'inference_only',
             'external_experiment', 'dropout', 'SGDnesterov','bipolarize_prototypes', 'metatrain_frozen', 'save_confusion', 
             'advance_augment', 'use_etf_prototypes']
    integer_tuples = ['image_size', 'num_filters', 'kernel_sizes', 'maxpool_sizes', 'dense_sizes', 'scheduler_steps']
    float_tuples = ['dataset_split']
            

    if key in integers:
        value = int(value)
    elif key in floats:
        value = float(value)
    elif key in strings:
        pass
    elif key in bools:
        value = parse_bool(value)
    elif key in integer_tuples:
        value = literal_eval(value)
    elif key in float_tuples:
        value = literal_eval(value)
    else:
        raise KeyError("Parameter key case for \'{}\' not covered.".format(key))
    return key, value


def parse_bool(value):
    if value in ['True', 'true', 't', '1']:
        value = True
    elif value in ['False', 'false', 'f', '0']:
        value = False
    else:
        raise ValueError("Boolean value not recognized.")
    return value


def parse_answer(value):
    if value in ['Yes', 'yes', 'Y', 'y']:
        value = True
    elif value in ['No', 'no', 'N', 'n']:
        value = False
    else:
        raise ValueError("Boolean value not recognized.")

    return value


def create_log_str(args):
    # Log directory
    log_prefix = args.logprefix if args.logprefix \
        else '/dataP/man/log/test'
    log_suffix = args.logsuffix if args.logsuffix \
        else datetime.datetime.now().strftime('%Y_%m_%d/%H_%M_%S')
    log_dir = args.logdir if args.logdir \
        else log_prefix + '/' + log_suffix
    return log_dir

def is_overwritten(key, parameters, defaults):
    return True if parameters[key] != defaults[key] else False

def round_to_pow2(x):
    """
    :param x:
    :type x: int
    :return:
    """
    return 1 << (x - 1).bit_length()

class ParamFSCIL():
    def __init__(self, args) -> None:
        # Define default self.parameters that are allowed to be called
        self.parameters = {
            # Architecture self.parameters
            ## Network
            'block_architecture':               'mini_resnet12', 
            'block_interface':                  'GAP_FC', # unused
            'resblocknorm':                     "none", # unused
            'num_filters':                      None,
            'kernel_sizes':                     None,
            'maxpool_sizes':                    None,
            'dense_sizes':                      (1024,), # unused
            'width_mult':                       1.0,
            'round_nearest':                    1,

            # Model
            'dim_features':                     512,
            'sharpening_activation':            'relu',  # 'softabs', 'softrelu', 'abs', 'relu', 'exp'
            'sharpening_strength':              10.,
            'representation':                   'real',
            'pretrainFC_vec':                   None,
            'ETFpoints':                        None, # obselete
        
            # Hardware approximations
            'normalize_weightings':             True,

            # Retraining operations
            'retrain_iter':                     0,
            'nudging_iter':                     0,
            'nudging_act':                      'doubleexp',
            'nudging_act_exp':                  4,
            'retrain_act':                      'tanh',
            'bipolarize_prototypes':            False,
            'use_etf_prototypes':               False,
            
            # Dataset self.parameters
            'dataset':                          'mini_imagenet',
            'data_folder':                      './data/miniimagenet',
            'image_size':                        (3,84, 84),
            'with_repetition':                  True, # unused
            'allow_empty_classes':              True, # unused
            'dataset_split':                    (0.85, 0.15), # unused
            'random_seed':                      None,
            'num_workers':                      4,

            # Optimization self.parameters
            'trainstage':                       'pretrain_baseFSCIL',
            'optimizer':                        'SGD', # unused
            'SGDmomentum':                      0.9,
            'SGDweight_decay':                  5e-4, 
            'SGDnesterov':                      True,
            'learning_rate':                    1e-4,
            'lr_step_size':                     30000,
            'norm_weight':                      10,
            'dropout':                          False,
            'dropout_rate':                     0.0,
            'dropout_rate_interm':              0.0,
            'pretrainFC':                       'linear',
            'metatrain_loss':                   'BCELoss',
            'pretrain_loss':                    'CrossEntropyLoss',
            'fscil_loss':                       'myCosineLoss',
            'loss_param':                       0.05, # for MultiMarginLoss with softab

            # Scheduler parameter
            'scheduler_type':                   "CosineAnnealing", # "CosineAnnealing" and "Step"
            'scheduler_warmup_step':            100,
            'scheduler_min_lr_scaler':          0.01, #only for "CosineAnnealing"
            'scheduler_gamma':                  0.25, #only for "Step"
            'scheduler_steps':                  [80, 120, 140, 160, 180], #only for "Step"
            'scheduler_start_lr_scaler':        0.25, #only for "Step"
            
            # Problem self.parameters
            'num_ways':                         5,
            'num_shots':                        1,
            'num_ways_training':                None,
            'num_shots_training':               None,
            'num_query_training':               None,
            'num_ways_inference':               None,
            'num_shots_inference':              None,
            'num_query_inference':              None,

            # Test/training self.parameters
            'batch_size':                       None,
            'batch_size_training':              None,
            'batch_size_inference':             None,
            'max_train_iter':                   30000,
            'max_val_iter':                     20, 
            'max_test_iter':                    1000,
            'validation_frequency':             500,
            'metatrain_frozen':                False,

            # Compression self.parameters
            'em_compression':                   'none', 
            'em_compression_nsup':              2,

            # Representation check self.parameters
            'check_representation_similarity':  False,
            'average_support_vector_inference': True,
            # Logging self.parameters
            'log_dir':                          None,
            'resume':                           '',
            'inference_only':                   False,
            'experiment_dir':                   None,
            'external_experiment':              False,
            'summary_frequency_seldom':         2500,
            'summary_frequency_often':          250,
            'summary_frequency_very_often':     10,
            'save_confusion':                   False,

            # For preassigned model, optimizer, scheduler
            "scheduler":                        None,
            "optim":                            None,
            "load_checkpoint":                  True,
            "model":                            None,
            # For quantization
            "quant_controller":                 None,
            "integerize":                       None,
            "quant_type":                       None,
            # For quantized last layer backprop
            "eps_proto":                        None,
            "last_layer":                       None,

            # For neural collapse
            "step_list":                        None,
            "copy_list":                        None,
            "advance_augment":                  False,
            "augments":                         None,
            
            # aux losses
            # orthogonal loss
            "lambda_ortho":                     0,
            # contrastive learning loss
            "lambdaCL":                         0,
            "typeCL":                           None, # "SimCLR", "SupCon", None
            "temperatureCL":                    0.07,
            "overlapCL":                        0,
        }

        # Dependent defaults
        self.parameters['batch_size'] = round_to_pow2(4 * self.parameters['num_ways'])

        #  Create the log directory path
        self.parameters['log_dir'] = create_log_str(args)

        # Store defaults to check what was overwritten
        self.defaults = deepcopy(self.parameters)

        # Check if parameter arguments are valid, cast the strings and update the self.parameters
        if args.parameter:
            for key, value in args.parameter:
                if key not in self.parameters:
                    raise KeyError('Not a valid parameter key: \"{}\".'.format(key))

                # Cast the strings and update the self.parameters
                key, value = test_and_cast(key, value)
                self.parameters.update({key: value})
        
        # # Set randomess
        # t.manual_seed(self.parameters['random_seed'])
        # random.seed(self.parameters['random_seed'])
        # np.random.seed(self.parameters['random_seed'])
        # t.cuda.manual_seed(self.parameters['random_seed'])
        # t.backends.cudnn.deterministic = True
        # t.backends.cudnn.benchmark = False

        # --------------------------------------------------------------------------------------------------
        # Dependent Parameter Updates
        # --------------------------------------------------------------------------------------------------

        self.parameters['experiment_dir'] = self.parameters['log_dir'] + '/experiment'

        if self.parameters['num_filters'] is None:
            self.parameters['num_filters'] = (64,160,320,640)

        if self.parameters['batch_size_training'] is None:
            if self.parameters['batch_size']:
                self.parameters['batch_size_training'] = self.parameters['batch_size']
            else:
                raise ValueError
        if self.parameters['batch_size_inference'] is None:
            if self.parameters['batch_size']:
                self.parameters['batch_size_inference'] = self.parameters['batch_size']
            else:
                raise ValueError
        if self.parameters['num_ways_training'] is None:
            if self.parameters['num_ways']:
                self.parameters['num_ways_training'] = self.parameters['num_ways']
            else:
                raise ValueError
        if self.parameters['num_ways_inference'] is None:
            if self.parameters['num_ways']:
                self.parameters['num_ways_inference'] = self.parameters['num_ways']
            else:
                raise ValueError
        if self.parameters['num_shots_training'] is None:
            if self.parameters['num_shots']:
                self.parameters['num_shots_training'] = self.parameters['num_shots']
            else:
                raise ValueError
        if self.parameters['num_shots_inference'] is None:
            if self.parameters['num_shots']:
                self.parameters['num_shots_inference'] = self.parameters['num_shots']
            else:
                raise ValueError
        if self.parameters['num_query_training'] is None:
            if self.parameters['batch_size_training']:
                self.parameters['num_query_training'] = self.parameters['batch_size_training']
            else: 
                raise ValueError
        if self.parameters['num_query_inference'] is None:
            if self.parameters['batch_size_inference']:
                self.parameters['num_query_inference'] = self.parameters['batch_size_inference']
            else: 
                raise ValueError
        
        # Remove unwanted self.parameters
        unwanted_parameters = ['batch_size', 'num_ways', 'num_shots']
        for unwanted_parameter in unwanted_parameters:
            self.parameters.pop(unwanted_parameter)

        # Set all unused self.parameters to None
        unused_parameters = []
        
        if not self.parameters['external_experiment']:
            unused_parameters += ['experiment_dir']

        if 'binary' not in self.parameters['representation']:
            unused_parameters += ['approximate_binary_similarity']

        if self.parameters['sharpening_activation'] not in ['softabs', 'softrelu','scaledexp']:
            unused_parameters += ['sharpening_strength']

        for unused_parameter in unused_parameters:
            self.parameters[unused_parameter] = None
        
        if self.parameters["dataset"] == "cifar100":
            if self.parameters["retrain_iter"] < 0:
                self.parameters["step_list"] = [0, 50, 75,   100, 120, 140,   160, 200, 200] 
            else:
                self.parameters["step_list"] = 9*[self.parameters["retrain_iter"]]
            self.parameters["copy_list"] = [1, 1, 1,   1, 1, 1,   1, 1, 1]
        elif self.parameters["dataset"] == "mini_imagenet":
            if self.parameters["retrain_iter"] < 0:
                self.parameters["step_list"] = [0, 100, 110,   120, 130, 140,   150, 160, 170]
            else:
                self.parameters["step_list"] = 9*[self.parameters["retrain_iter"]]
            self.parameters["copy_list"] = [1, 1, 2,   3, 4, 5,   6, 7, 8]
        elif self.parameters["dataset"] == "omniglot":
            if self.parameters["retrain_iter"] < 0:
                self.parameters["step_list"] = [0, 100, 110,   120, 130, 140,   150, 160, 170,  180]
            else:
                self.parameters["step_list"] = 10*[self.parameters["retrain_iter"]]
            self.parameters["copy_list"] = [1, 1, 2,   3, 4, 5,   6, 7, 8,  9]
        elif self.parameters["dataset"] == "cub200":
            if self.parameters["retrain_iter"] < 0:
                self.parameters["step_list"] = [0, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
            else:
                self.parameters["step_list"] = 11*[self.parameters["retrain_iter"]]
            self.parameters["copy_list"] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        if self.parameters["advance_augment"] == True:
            from lib.augment.augments import Augments
            from lib.augment.cutmix import BatchCutMixLayer
            from lib.augment.mixup import BatchMixupLayer
            from lib.augment.mixup import BatchMultiMixupLayer
            from lib.augment.idty import Identity
            augs = [
                BatchMixupLayer(alpha=0.8, prob=0.4),
                BatchCutMixLayer(alpha=1.0, prob=0.4),
                Identity(prob=0.2),
            ]
            self.parameters["augments"] = Augments(augs)