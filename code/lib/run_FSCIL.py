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
import csv
import datetime
from email.mime import base
import time
from copy import copy
from operator import itemgetter

import shutil
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import numpy as np
from dotmap import DotMap

import progressbar
import tqdm

from lib.model import *
# from lib.util import csv2dict, loadmat
from lib.torch_blocks import myCosineLoss, CustomNLLLoss, SupConLoss, MSELoss, GroupCELoss, NegativeCELoss
from plot.confusion_support import plot_confusion_support, avg_sim_confusion, plot_confusion_meta
import os.path
import pdb

from lib.dataloader.FSCIL.data_utils import *
from .scheduler import CosineAnnealingWarmupRestarts, StepWarmupRestarts

def pretrain_baseFSCIL(verbose,**parameters):
    '''
    Pre-training on base session
    ''' 
    args = DotMap(parameters)
    args.gpu = 0 
    
    # Initialize the dataset generator and the model
    args = set_up_datasets(args)
    trainset, train_loader, val_loader = get_base_dataloader(args)

    model = KeyValueNetwork(args, mode="pretrain")
    if isinstance(args.model, KeyValueNetwork):
        model = args.model
    model.mode = 'pretrain' 

    if args.typeCL is not None:
        _, train_cl_loader, _ =  get_base_dataloader_contrastive(args)
        in_feat = model.embedding.fc.in_features
        contrastive_head = ContrastiveHead(
            in_feat=in_feat, out_feat=args.dim_features, 
            mid_feat=args.dim_features, overlap=args.overlapCL)
        criterionCL = SupConLoss(temperature=args.temperatureCL)
        if args.gpu is not None:
            contrastive_head = contrastive_head.cuda(args.gpu)
            criterionCL = criterionCL.cuda(args.gpu)

    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            print("\t{}".format(key).ljust(40) + "{}".format(value))

    writer = SummaryWriter(args.log_dir)

    # Take start time
    start_time = time.time()

    if args.pretrain_loss == 'NLLLoss':
        criterion = nn.NLLLoss()
    elif args.pretrain_loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.pretrain_loss == 'CustomNLLLoss':
        criterion = CustomNLLLoss()
    elif args.pretrain_loss == 'MultiMarginLoss':
        criterion = nn.MultiMarginLoss(margin = args.loss_param) 
    elif args.pretrain_loss == 'MSELoss':
        criterion = MSELoss(multiplier= args.loss_param) 

    if args.gpu is not None: 
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
       
    if args.optim is not None:
        optimizer = args.optim
    else :
        optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate,nesterov=args.SGDnesterov, 
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)
        
        if args.pretrainFC == "rotatedetf":
            model.fc_pretrain[2].orthonormal.requires_grad = True
            optimizer.add_param_group({"params": model.fc_pretrain[2].orthonormal,
                        "lr":args.learning_rate,"nesterov":args.SGDnesterov, 
                        "weight_decay":0, "momentum":args.SGDmomentum})

    if args.scheduler is not None:
        scheduler = args.scheduler
    else:
        if args.scheduler_type == "CosineAnnealing":
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.max_train_iter*len(train_loader), cycle_mult=1.0, 
                                                    max_lr=args.learning_rate, min_lr=args.learning_rate*args.scheduler_min_lr_scaler, 
                                                    warmup_steps=args.scheduler_warmup_step, gamma=0.0)
        elif args.scheduler_type == "Step":
            scheduler = StepWarmupRestarts(optimizer, steps=args.scheduler_steps, gamma=args.scheduler_gamma,
                                        warmup_start_lr_scale=args.scheduler_start_lr_scaler, warmup_steps=args.scheduler_warmup_step, 
                                        len_loader=len(train_loader))
        elif args.scheduler_type == "StepNoWarmup":
            scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)
    
    best_acc1 = 0
    # summary(model.embedding, (3, 32, 32))
    for epoch in tqdm.tqdm(range(0,args.max_train_iter),desc='Epoch'):
        if args.quant_controller is not None:
            for c in args.quant_controller:
                # mandatory for the begining of every epoch
                c.step_pre_training_epoch(epoch)

        if args.typeCL is not None:
            train_cl_iter = iter(train_cl_loader)

        losses = AverageMeter('Loss')
        acc = AverageMeter('Acc@1')
        
        model.train(True)

        for i, batch in enumerate(train_loader):
            if args.quant_controller is not None:
                for c in args.quant_controller:
                    c.step_pre_training_batch()
            
            data, train_label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]

            # Main loss function
            optimizer.zero_grad()
            if args.augments is not None:
                data, lam, gt_label, gt_label_aux = args.augments(data, train_label)
                
                output = model(data)
                if gt_label_aux is not None:
                    loss1 = criterion(output,gt_label)
                    loss2 = criterion(output,gt_label_aux)
                    loss = lam * loss1 + (1-lam) * loss2
                else :
                    loss = criterion(output,gt_label)
            else:
                output = model(data) 
                loss = criterion(output,train_label)
            
            # feature orthogonality loss
            if args.lambda_ortho != 0:
                proto = F.normalize(model.proto, dim=0, p=2)
                loss_reg = proto.t() @ proto - t.eye(proto.shape[1], device=proto.device)
                loss_reg = torch.mean(loss_reg*loss_reg)
                loss = loss + args.lambda_ortho*loss_reg

            # contrastive learning loss
            if (args.typeCL is not None) and (args.lambdaCL != 0):
                images_cl, labels_cl = next(train_cl_iter)
                
                if args.gpu is not None:
                    images_cl[0] = images_cl[0].cuda(args.gpu, non_blocking=True)
                    images_cl[1] = images_cl[1].cuda(args.gpu, non_blocking=True)
                    labels_cl = labels_cl.cuda(args.gpu, non_blocking=True)
                images_cl = torch.cat([images_cl[0], images_cl[1]], dim=0)
                bsz = labels_cl.shape[0]
                
                
                if hasattr(model.embedding, "forward_conv"):
                    proto_lv1 = model.embedding.forward_conv(images_cl)
                else:
                    proto_lv1 = model.embedding.conv_embedding(images_cl)
                proto_lv2 = model.embedding.fc(proto_lv1)

                # features = F.normalize(contrastive_head(proto_lv2, proto_lv1), dim=1)
                # # DEBUG
                features = F.normalize(model.fc_pretrain[1](model.fc_pretrain[0](proto_lv2)) @ model.fc_pretrain[2].etf_null, dim=1)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                if args.typeCL == 'SupCon':
                    loss_cl = criterionCL(features, labels_cl)
                elif args.typeCL == 'SimCLR':
                    loss_cl = criterionCL(features)
                else:
                    raise ValueError('contrastive method not supported: {}'.
                                    format(args.typeCL))
                loss = loss + args.lambdaCL*loss_cl
                
            # Backpropagation
            loss.backward()
            optimizer.step()

            accuracy = top1accuracy(output.argmax(dim=1),train_label)

            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(), data.size(0))
            
            scheduler.step()

        # write to tensorboard
        writer.add_scalar('training_loss/pretrain_CEL',losses.avg,epoch)
        writer.add_scalar('accuracy/pretrain_train',acc.avg, epoch)

        val_loss, val_acc_mean,_ = validation(model,criterion,val_loader, args)
        writer.add_scalar('validation_loss/pretrain_CEL', val_loss,epoch)
        writer.add_scalar('accuracy/pretrain_val', val_acc_mean,epoch)

        is_best = val_acc_mean > best_acc1
        best_acc1 = max(val_acc_mean, best_acc1)

        args_dict = args.toDict()
        args_dict.pop("Dataset")
        args_dict.pop("scheduler")
        args_dict.pop("optim")
        args_dict.pop("model")
        save_checkpoint({
            'args': args_dict,
            'train_iter': epoch + 1,
            'arch': args.block_architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,savedir=args.log_dir) 

    writer.close()

def metatrain_baseFSCIL(verbose,**parameters):
    '''
    Meta-training on base session
    ''' 

    # Argument Preparation
    args = DotMap(parameters)
    args.gpu = 0 
    
    # Initialize the dataset generator and the model
    args = set_up_datasets(args)
    trainset, train_loader, val_loader = get_base_dataloader_meta(args)

    model = KeyValueNetwork(args,mode='meta')
    if isinstance(args.model, KeyValueNetwork):
        model = args.model

    model.mode = 'meta'
    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            print("\t{}".format(key).ljust(40) + "{}".format(value))

    writer = SummaryWriter(args.log_dir)

        
    # Take start time
    start_time = time.time()
    
    # Choose loss function
    base_criterion = None
    criterion = nn.BCELoss() # Default
    if args.metatrain_loss == 'NLLLoss':
        base_criterion = nn.NLLLoss()
    elif args.metatrain_loss == 'CrossEntropyLoss':
        base_criterion = nn.CrossEntropyLoss()
    elif args.metatrain_loss == 'CustomNLLLoss':
        base_criterion = CustomNLLLoss()
    elif args.metatrain_loss == 'MultiMarginLoss':
        base_criterion = nn.MultiMarginLoss(margin = args.loss_param) # margin dependent on activation!
    elif args.pretrain_loss == 'MSELoss':
        criterion = MSELoss(multiplier= args.loss_param) 

    if args.gpu is not None: 
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        if base_criterion:
            base_criterion = base_criterion.cuda(args.gpu)

    if args.metatrain_loss in ["NLLLoss", "CrossEntropyLoss", "CustomNLLLoss", "MultiMarginLoss", "MSELoss"]: # Wrapper to convert back from one-hot
        criterion = lambda output, query_label : base_criterion(output, torch.max(query_label, 1)[1]) 
    
    if args.optim is not None:
        optimizer = args.optim
    else :
        optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate,nesterov=args.SGDnesterov, 
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)
    
    if args.scheduler is not None:
        scheduler = args.scheduler
    else:
        if args.scheduler_type == "CosineAnnealing":
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.max_train_iter, cycle_mult=1.0, 
                                                    max_lr=args.learning_rate, min_lr=args.learning_rate*args.scheduler_min_lr_scaler, 
                                                    warmup_steps=args.scheduler_warmup_step, gamma=0.0)
        elif args.scheduler_type == "Step":
            scheduler = StepWarmupRestarts(optimizer, steps=args.scheduler_steps, gamma=args.scheduler_gamma,
                                        warmup_start_lr_scale=args.scheduler_start_lr_scaler, warmup_steps=args.scheduler_warmup_step, 
                                        len_loader=len(train_loader))
        elif args.scheduler_type == "StepNoWarmup":
            scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=args.scheduler_gamma)
    
    if args.load_checkpoint:
        model,optimizer,scheduler, start_train_iter,best_acc1 = load_checkpoint(model,optimizer,scheduler,args)
    else:
        best_acc1 = 0
        start_train_iter = 0
    best_acc1 = 0
    k = args.num_ways_training*args.num_shots_training

    losses = AverageMeter('Loss')
    acc = AverageMeter('Acc@1')
    train_iterator = iter(train_loader) 

    # Freeze model
    if args.metatrain_frozen:
        print("Freezing Weights!")
        for name, param in model.named_parameters():
            print(name)
            param.requires_grad = False

        # Unfreeze fc layers
        print("Unfrozen layers:")
        for name, param in model.embedding.fc.named_parameters():
            print(name)
            param.requires_grad = True
    
    summary(model.embedding, (3, 32, 32))
    print("Loss function: " + args.metatrain_loss)

    for i in tqdm.tqdm(range(start_train_iter,args.max_train_iter), initial=start_train_iter, total=args.max_train_iter,desc='Epoch'):
        if args.quant_controller is not None:
            for c in args.quant_controller:
                c.step_pre_training_epoch(i)
                c.step_pre_training_batch()
        batch = next(train_iterator)
        data, train_label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]
        train_label_onehot = F.one_hot(train_label, num_classes = args.num_classes).float()
        proto, query = data[:k], data[k:]
        proto_label, query_label = train_label_onehot[:k], train_label_onehot[k:]
        model.eval()
        with t.no_grad():
            model.write_mem(proto, proto_label)
        
         # forward pass
        model.train()
        optimizer.zero_grad()
        output = model(query)

        loss = criterion(output, query_label) # Training loss

        if args.lambda_ortho != 0:
            proto = F.normalize(model.proto, dim=0, p=2)
            loss_reg = proto.t() @ proto - t.eye(proto.shape[1], device=proto.device)
            loss_reg = torch.mean(loss_reg*loss_reg)
            loss = loss + args.lambda_ortho*loss_reg

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Do evaluation           
        predicted_labels, predicted_certainties, actual_labels, actual_certainties, accuracy = process_result(
                output,query_label)

        scheduler.step()


        if not i % args.summary_frequency_very_often:
            # write to tensorboard
            writer.add_scalar('training_loss/log_loss',loss.item(),i)
            writer.add_scalar('accuracy/training',accuracy*100,i)
            print("\tTraining loss: " + str(loss.item()) + "\tTraining accuracy: " + str(accuracy.item()*100))

        if not i % args.validation_frequency: 
            val_loss, val_acc_mean = validation_onehot(model,criterion,val_loader,args,
                                                    num_classes = args.num_classes)
            writer.add_scalar('validation_loss/log_loss', val_loss,i)
            writer.add_scalar('accuracy/validation', val_acc_mean,i)

            is_best = val_acc_mean > best_acc1
            best_acc1 = max(val_acc_mean, best_acc1)

            args_dict = args.toDict()
            args_dict.pop("Dataset")
            args_dict.pop("scheduler")
            args_dict.pop("optim")
            args_dict.pop("model")
            save_checkpoint({
                'args': args_dict,
                'train_iter': i + 1,
                'arch': args.block_architecture,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best,savedir=args.log_dir) 
        
            # Confusion matrix        
            if (args.save_confusion):
                # conf_fig = plot_confusion_meta(model, train_loader, args.num_ways_training) # Training
                conf_fig = plot_confusion_meta(model, val_loader, args.num_ways_training) # Validation
                writer.add_figure('Metatrain confusion matrix (validation)',conf_fig, i)

    args_dict = args.toDict()
    args_dict.pop("Dataset")
    args_dict.pop("scheduler")
    args_dict.pop("optim")
    args_dict.pop("model")
    save_checkpoint({
        'args': args_dict,
        'train_iter': i + 1,
        'arch': args.block_architecture,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
    }, is_best,savedir=args.log_dir) 
    
    writer.close()

def train_FSCIL(verbose=False, **parameters):
    '''
    Main FSCIL evaluation on all sessions
    ''' 
    args = DotMap(parameters) 
    args = set_up_datasets(args)
    args.gpu = 0
    
    if args.model is not None:
        model = args.model
    else:
        model = KeyValueNetwork(args,mode='meta')

    # Store all parameters in a variable
    parameters_list, parameters_table = process_dictionary(parameters)

    # Print all parameters
    if verbose:
        print("Parameters:")
        for key, value in parameters_list:
            print("\t{}".format(key).ljust(40) + "{}".format(value))

    # Write parameters to file
    if not args.inference_only:
        filename = args.log_dir + '/parameters.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        #retrain
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            keys, values = zip(*parameters_list)
            writer.writerow(keys)
            writer.writerow(values)

    writer = SummaryWriter(args.log_dir)

    # Take start time
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()

    if args.gpu is not None: 
        t.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    # set all parameters except FC to trainable false
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model.embedding, "fc"):
        for param in model.embedding.fc.parameters():
            param.requires_grad = True
    
    if args.optim is not None:
        optimizer = args.optim
    else :
        optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate,nesterov=args.SGDnesterov, 
                            weight_decay=args.SGDweight_decay, momentum=args.SGDmomentum)
    
    
    if args.scheduler is not None:
        scheduler = args.scheduler
    else:
        scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)
    
    if args.load_checkpoint:
        model,optimizer,scheduler, start_train_iter,best_acc1 = load_checkpoint(model,optimizer,scheduler,args)

    if hasattr(model, "on_board"):
        model.reset_activation()
        model.init_eps()
        model.init_prototypes()
        print("Reset Done")

    for session in range(args.sessions): 
        nways_session = args.base_class + session*args.way
        
        train_set, train_loader, test_loader = get_dataloader(args, session)
        # update model
        batch = next(iter(train_loader))

        if hasattr(model, "on_board"):
            align_on_board(model,batch, optimizer,args,writer,session,nways_session)
        elif (args.last_layer is not None) and (args.eps_proto is not None):
            align_integer(model,batch, optimizer,args,writer,session,nways_session,args.last_layer,args.eps_proto)
        else:
            align(model,batch, optimizer,args,writer,session,nways_session)        
        
        loss, acc, conf_fig = validation(model,criterion,test_loader,args,nways_session)
        print("Session {:}: {:.2f}%".format(session+1,acc))

        writer.add_scalar('accuracy/cont', acc, session)

        # Plot confusion matrix
        if (args.save_confusion):
            proto_fig = plot_confusion_support(model.key_mem.data.cpu(),savepath="{:}/session{:}".format(args.log_dir,session))
            writer.add_figure('prototype_sim',proto_fig,session)

    writer.close()

def align(model,data,optimizer,args,writer,session,nways_session): 
    '''
    Alignment of FC using MSE Loss and feature replay
    '''

    losses = AverageMeter('Loss')
    criterion = myCosineLoss(args.retrain_act)
    dataset = myRetrainDataset(data[0], data[1])
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch_size_training)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer,step_size=args.lr_step_size,gamma=0.1)

    # Stage 1: Compute feature representation of new data
    model.eval()
    with t.no_grad():
        for x,target in dataloader: 
            x = x.cuda(args.gpu,non_blocking=True)
            if args.integerize is not None:
                x = args.integerize(x) 
            target = target.cuda(args.gpu,non_blocking=True)    
            model.update_feat_replay(x, target)

    # Stage 2: Compute prototype based on GAAM
    feat, label = model.get_feat_replay()
    if args.quant_type=="true":
        feat = torch.trunc(feat)
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat,label,nways_session)

    # Stage 3: Nuddging
    model.nudge_prototypes(nways_session,writer,session,args.gpu)

    # Bipolarize prototypes in Mode 2
    if args.bipolarize_prototypes:
        target = model.bipolarize_prototypes()
    elif args.use_etf_prototypes:
        target = model.use_etf_prototypes()
    else:
        target = model.key_mem.data

    # Stage 4: Update Retraining the FC
    model.embedding.fc.train()
    for epoch in range(args.step_list[session]):
        optimizer.zero_grad()
        support = model.get_support_feat(feat)
        loss = criterion(support[:nways_session],target[:nways_session])

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('retraining/loss_sess{:}'.format(session), loss.item(), epoch)

    # Stage 5: Fill up prototypes again
    model.eval() 
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat,label,nways_session)

    # Stage 6: Optional EM compression
    if args.em_compression == "hrr": 
        model.hrr_superposition(nways_session, args.em_compression_nsup)

def align_on_board(model,data,optimizer,args,writer,session,nways_session): 
    '''
    Alignment of FC using MSE Loss and feature replay
    '''
    dataset = myRetrainDataset(data[0], data[1])
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch_size_training)

    # Stage 1: Compute feature representation of new data
    model.eval()
    with t.no_grad():
        for x,target in dataloader:
            x = x.cuda(args.gpu,non_blocking=True)
            if args.integerize is None:
                print("intigerize function for on board testing")
                assert False
            x = args.integerize(x) 
            target = target.cuda(args.gpu,non_blocking=True)
            model.update_feat_replay(x, target)

    # Stage 2: Compute prototype based on GAAM
    if model.mode != "neurcollapse":
        model.recalculate_prototypes_feat()

    # # Stage 3: Nuddging (not supported yet for on board training) TODO
    # model.nudge_prototypes(nways_session,writer,session,args.gpu)

    # Bipolarize prototypes in Mode 2
    if args.bipolarize_prototypes:
        model.bipolarize_prototypes()

    # Stage 4: Update Retraining the FC
    for epoch in range(args.step_list[session]):
        loss = model.last_layer_training(learn_rate=optimizer.param_groups[0]['lr'], epoch=1)
        writer.add_scalar('retraining/loss_sess{:}'.format(session), loss.item(), epoch)

    # Stage 5: Fill up prototypes again
    if model.mode != "neurcollapse":
        model.recalculate_prototypes_feat()

# only support PACTLinear layer in last layer
def align_integer(model,data,optimizer,args,writer,session,nways_session,last_layer,eps_proto): 
    
    '''
    Alignment of FC using MSE Loss and feature replay
    '''

    losses = AverageMeter('Loss')
    criterion = myCosineLoss(args.retrain_act)
    dataset = myRetrainDataset(data[0], data[1])
    dataloader = DataLoader(dataset=dataset, batch_size = args.batch_size_training)

    # Stage 1: Compute feature representation of new data
    model.eval()
    with t.no_grad():
        for x,target in dataloader: 
            x = x.cuda(args.gpu,non_blocking=True)
            if args.integerize is not None:
                x = args.integerize(x) 
            target = target.cuda(args.gpu,non_blocking=True)    
            model.update_feat_replay(x, target)

    # Stage 2: Compute prototype based on GAAM
    feat, label = model.get_feat_replay()
    if args.quant_type=="true":
        feat = torch.trunc(feat) #trunc is basically floor
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat,label,nways_session)

    # # Stage 3: Nuddging
    # # Nudging is still not integerized
    # model.nudge_prototypes(nways_session,writer,session,args.gpu)

    # Initialize important variable for integer backprop
    dev = next(model.parameters()).device
    eps_in  = torch.tensor(eps_proto[0], device=dev)
    eps_out = torch.tensor(eps_proto[1], device=dev)
    eps_w   = torch.tensor(eps_proto[2], device=dev)
    eps_b = eps_out
    eps_gb = eps_b # epsilon gradient b
    eps_gw = eps_in * eps_out # epsilon gradient w

    # Training functions for True Quantized Network
    if args.quant_type=="true":
        w = t.clone(last_layer.weight.data)
        b = t.clone(last_layer.bias.data)

        # Bipolarize prototypes in Mode 2
        sign_value = torch.round(1/eps_out)
        if args.bipolarize_prototypes:
            target = t.sign(model.key_mem.data)*sign_value
        if args.use_etf_prototypes:
            target = torch.round(model.fc_pretrain[2].get_etf()/eps_out).t()

        def cossim_grad(x, target, eps_x):
            mag_x = t.linalg.norm(x,ord=2,dim=1,keepdim=True)
            mag_t = t.linalg.norm(target,ord=2,dim=1,keepdim=True)
            cossim = (x*target).sum(1,keepdim=True)/(mag_x*mag_t)
            # Note the sign of cossim grad is reversed because
            # we want to maximize cossim grad (dont miimize it)
            grad_float = -target/(mag_x*mag_t) + x*(cossim/(mag_x*mag_x))
            return torch.trunc(grad_float/(eps_x*eps_x))
        
        def value_update(x, eps_x, gx, eps_gx, learn_rate, remainder=None, momentum=None, beta_mul=9, beta_div=10):
            bat_size = gx.shape[0]
            gx_sum = gx.sum(0)
            gx_div = (torch.round(eps_x/(eps_gx*learn_rate))*bat_size)
            if momentum is not None:
                momentum = torch.trunc((momentum*beta_mul+gx_sum*(beta_div-beta_mul))/beta_div)
                x_delta = torch.trunc(momentum/gx_div)
                return (x - x_delta), momentum
            elif remainder is not None:
                x_delta = torch.round((gx_sum + remainder)/gx_div)
                remainder = (gx_sum + remainder) - x_delta*gx_div
                return (x - x_delta), remainder
            else:
                x_delta = torch.round(gx_sum/gx_div)
                return (x - x_delta)
    
    # Training functions for Fake Quantized Network
    else :
        # quantise value
        w = t.clone(last_layer.weight_q)
        b = t.clone(last_layer.get_bias_q(eps_in))

        # Bipolarize prototypes in Mode 2
        sign_value = torch.round(1/eps_out)*eps_out
        if args.bipolarize_prototypes:
            target = t.sign(model.key_mem.data)*sign_value
        if args.use_etf_prototypes:
            target = (torch.round(model.fc_pretrain[2].get_etf()/eps_out)*eps_out).t()
        
        def cossim_grad(x, target, eps_x):
            mag_x = t.linalg.norm(x,ord=2,dim=1,keepdim=True)
            mag_t = t.linalg.norm(target,ord=2,dim=1,keepdim=True)
            cossim = (x*target).sum(1,keepdim=True)/(mag_x*mag_t)
            # Note the sign of cossim grad is reversed because
            # we want to maximize cossim grad (dont miimize it)
            grad_float = -target/(mag_x*mag_t) + x*(cossim/(mag_x*mag_x))
            return torch.trunc(grad_float/eps_x)*eps_x
        
        def value_update(x, eps_x, gx, eps_gx, learn_rate, remainder=None, momentum=None, beta_mul=9, beta_div=10):
            bat_size = gx.shape[0]
            gx_div = (torch.round(eps_x/(eps_gx*learn_rate))*bat_size)*(eps_gx/eps_x)
            gx_sum = gx.sum(0)
            if momentum is not None:
                momentum = torch.trunc((momentum*beta_mul+gx_sum*(beta_div-beta_mul))/(beta_div*eps_gx))*eps_gx
                x_delta = momentum/gx_div
                return (x - torch.trunc(x_delta/eps_x)*eps_x), momentum
            elif remainder is not None:
                x_delta = torch.round((gx_sum + remainder)/(gx_div*eps_x))
                remainder = (gx_sum + remainder) - x_delta*(gx_div*eps_x)
                return (x - x_delta*eps_x), remainder
            else:
                x_delta = gx_sum/gx_div
                return (x - torch.round(x_delta/eps_x)*eps_x)


    # Stage 4: Update Retraining the FC
    model.embedding.fc.train()
    momentum_b = 0
    momentum_w = 0

    for epoch in range(args.step_list[session]):
        with torch.no_grad():
            learn_rate = optimizer.param_groups[0]['lr'] * (0.1**(epoch//args.lr_step_size)) # with LRStep-like scheduler
            support = feat @ w.transpose(0,1) + b[None,:] #get_support_feat
            grad_b = cossim_grad(support[:nways_session], target[:nways_session], eps_out)
            grad_w = t.bmm(grad_b.unsqueeze(2),feat[:nways_session].unsqueeze(1))
            # b = value_update(b, eps_b, grad_b, eps_gb, learn_rate) # without momentum neither remainder
            # w = value_update(w, eps_w, grad_w, eps_gw, learn_rate) # without momentum neither remainder
            b, momentum_b = value_update(b, eps_b, grad_b, eps_gb, learn_rate, remainder=momentum_b) #with remainder
            w, momentum_w = value_update(w, eps_w, grad_w, eps_gw, learn_rate, remainder=momentum_w) #with remainder
            # b, momentum_b = value_update(b, eps_b, grad_b, eps_gb, learn_rate, momentum=momentum_b) #with momentum
            # w, momentum_w = value_update(w, eps_w, grad_w, eps_gw, learn_rate, momentum=momentum_w) #with momentum

            loss = criterion(support[:nways_session],model.key_mem.data[:nways_session])
            
            writer.add_scalar('retraining/loss_sess{:}'.format(session), loss.item(), epoch)

    last_layer.weight.data = w
    last_layer.bias.data = b

    # Stage 5: Fill up prototypes again
    model.eval() 
    model.reset_prototypes(args)
    model.update_prototypes_feat(feat,label,nways_session)

    # Stage 6: Optional EM compression
    if args.em_compression == "hrr": 
        model.hrr_superposition(nways_session, args.em_compression_nsup)

def validation(model,criterion,dataloader, args,nways_session=None):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    sim_conf = avg_sim_confusion(args.num_classes,nways_session)
    model.eval()
    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]
            if args.integerize is not None:
                data = args.integerize(data) 
            # from IPython import embed; embed()
            output = model(data)
            loss = criterion(output,label)
            accuracy = top1accuracy(output.argmax(dim=1),label)

            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item(),data.size(0))
            if nways_session is not None: 
                sim_conf.update(model.similarities.detach().cpu(),
                            F.one_hot(label.detach().cpu(), num_classes = args.num_classes).float())
    # Plot figure if needed
    fig = sim_conf.plot() if nways_session is (not None) else None
    return losses.avg, acc.avg, fig

def validation_onehot(model,criterion,dataloader, args, num_classes):
    #  

    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    model.eval()

    with t.no_grad(): 
        for i, batch in enumerate(dataloader):
            data, label = [_.cuda(args.gpu,non_blocking=True) for _ in batch]
            label = F.one_hot(label, num_classes = num_classes).float()

            output = model(data)

            loss = criterion(output,label)
            
            _, _, _, _, accuracy = process_result(
                output,label)

            losses.update(loss.item(),data.size(0))
            acc.update(accuracy.item()*100,data.size(0))
    
    return losses.avg, acc.avg

# --------------------------------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------------------------------
def process_result(predictions, actual):
    predicted_labels = t.argmax(predictions, dim=1)
    actual_labels = t.argmax(actual, dim=1)

    accuracy = predicted_labels.eq(actual_labels).float().mean(0,keepdim=True)
    # TBD implement those uncertainties
    predicted_certainties =0#
    actual_certainties = 0 #
    return predicted_labels, predicted_certainties, actual_labels, actual_certainties, accuracy


def process_dictionary(dict):
    # Convert the dictionary to a sorted list
    dict_list = sorted(list(dict.items()))

    # Convert the dictionary into a table
    keys, values = zip(*dict_list)
    values = [repr(value) for value in values]
    dict_table = np.vstack((np.array(keys), np.array(values))).T

    return dict_list, dict_table

# --------------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',savedir=''):
    t.save(state, savedir+'/'+filename)
    if is_best:
        shutil.copyfile(savedir+'/'+filename, savedir+'/'+'model_best.pth.tar')



def load_checkpoint(model,optimizer,scheduler,args):        
    # First priority: load checkpoint from log_dir 
    if os.path.isfile(args.log_dir+ '/checkpoint.pth.tar'):
        resume = args.log_dir+ '/checkpoint.pth.tar'
        print("=> loading checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = int(checkpoint['train_iter']) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc1 = checkpoint['best_acc1'] 
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))
    # Second priority: load from pretrained model
    # No scheduler and no optimizer loading here.  
    elif os.path.isfile(args.resume+'/checkpoint.pth.tar'):
        resume = args.resume+'/checkpoint.pth.tar'
        print("=> loading pretrain checkpoint '{}'".format(resume))
        if args.gpu is None:
            checkpoint = t.load(resume)
        else:
            # Map model to be loaded to specified single args.gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = t.load(resume, map_location=loc)
        start_train_iter = 0 
        best_acc1 = 0
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained checkpoint '{}' (train_iter {})"
              .format(args.log_dir, checkpoint['train_iter']))
    else:
        start_train_iter=0
        best_acc1 = 0
        print("=> no checkpoint found at '{}'".format(args.log_dir))
        print("=> no pretrain checkpoint found at '{}'".format(args.resume))
    
    return model, optimizer, scheduler, start_train_iter, best_acc1


# --------------------------------------------------------------------------------------------------
# Some Pytorch helper functions (might be removed from this file at some point)
# --------------------------------------------------------------------------------------------------

def convert_toonehot(label): 
    '''
    Converts index to one-hot. Removes rows with only zeros, such that 
    the tensor has shape (B,num_ways)
    '''
    label_onehot = F.one_hot(label)
    label_onehot = label_onehot[:,label_onehot.sum(dim=0)!=0]
    return label_onehot.type(t.FloatTensor)

def top1accuracy(pred, target):
    """Computes the precision@1"""
    batch_size = target.size(0)

    correct = pred.eq(target).float().sum(0)
    return correct.mul_(100.0 / batch_size)


class myRetrainDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
       
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
