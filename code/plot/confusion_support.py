
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

import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import pdb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def plot_confusion_meta(model, loader, nways):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    with t.no_grad():
        for inputs, labels in loader:
            output = model(inputs.to("cuda"))  # Feed Network
            output = (t.max(t.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # save prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth    
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    # classes = range(cf_matrix.shape[0])
    text = "Class: accuracy"
    offset = int(nways/2)
    diagonals = np.diagonal(cf_matrix)
    for i, acc in enumerate(diagonals[:offset]):
        text = text + "\n{:<4}: {:10.4f}".format(i, acc*100)
        if i + offset < nways: 
            text += "     {:<4}: {:10.4f}".format(i+offset, diagonals[i+offset]*100)


    fig = plt.figure(figsize=(16, 16))
    sn.heatmap(cf_matrix)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.annotate(text, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')    
    fig.tight_layout()
    return fig


def plot_confusion_support(prototypes, savepath=None):
    ''' 
    Parameters:
        prototypes: torch tensor (ways, dim_features)

    Returns:
        fig: pyplot figure 

    '''
    cm = get_confusion(prototypes).numpy()
    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, vmin=-1, vmax=1,
               cmap='seismic')
    
    fig.set_tight_layout(True)
    fig.colorbar(im)
    if savepath is not None: 
        fig.savefig(savepath+".pdf")
        np.savez(savepath+".npz", cm = cm)
    return fig


def get_confusion(support): 

    nways = support.shape[0]
    cm = t.zeros(nways,nways)
    cos = t.nn.CosineSimilarity()
    for way in range(nways): 
        cm[way] = cos(support[way:way+1],support)

    return cm

class avg_sim_confusion: 

    def __init__(self,nways,nways_session): 
        self.confusion_sum = t.zeros(nways,nways)
        self.nways_session = nways_session
        eps = 1e-8
        self.cnt = t.ones(1,nways)*eps

    def update(self,sim,onehot_label): 
        '''
        Parameters 
        ----------
        sim: Tensor (B,n_ways)
        onehot_label: Tensor (B,n_ways)
        '''
        acos_sim = t.acos(sim[:,:self.nways_session])
        self.confusion_sum[:self.nways_session] +=  t.matmul(t.transpose(acos_sim,0,1),onehot_label)
        self.cnt += t.sum(onehot_label,dim=0, keepdim=True)

    def plot(self):        
        cm = (self.confusion_sum/(self.cnt+1e-8))
        cm_diag = t.diagonal(cm).unsqueeze(0)
        interf_risk = cm_diag*t.div(1,cm+1e-8)
        mask = t.eye(interf_risk.shape[0],interf_risk.shape[1]).bool()
        interf_risk.masked_fill_(mask, 0)
        interf_risk[self.nways_session:]=0

        np.set_printoptions(precision=2)
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,7)) 
        ax1.imshow(t.transpose(cm,1,0), vmin=0, vmax=3.14,
               cmap='Blues')
        ax1.set_xlabel("Class vector")
        ax1.set_ylabel("Class data")

        ax2.imshow(t.transpose(interf_risk,1,0), vmin=0, vmax=1.5,
               cmap='Reds')
        ax2.set_xlabel("Class vector")
        ax2.set_ylabel("Class data")

        fig.set_tight_layout(True)

        return fig 
