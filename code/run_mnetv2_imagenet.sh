#!/bin/bash


EXP_ID="003"
SAVE_CONFUSION=True
CUDA_DEVICES='0'
DATASET=mini_imagenet
MODEL=mnetv2_x4

DIMFEATURES=512
WIDTH_MULT=1
ROUND_NEAREST=1
WAYS=60
BATCH_TRAIN=10
RANDOM_SEED=8

# Pretraining
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/pretrain_basetrain" \
 -p max_train_iter 500 -p data_folder "data" -p trainstage pretrain_baseFSCIL -p dataset $DATASET \
 -p random_seed $RANDOM_SEED -p learning_rate 0.05 -p batch_size 128 -p optimizer SGD \
 -p SGDnesterov True -p representation real -p dim_features $DIMFEATURES -p block_architecture $MODEL \
 -p SGDweight_decay 0.0005 -p SGDmomentum 0.9 -p advance_augment True -p scheduler_start_lr_scaler 0.25 \
 -p scheduler_type Step -p scheduler_warmup_step 3000 -p scheduler_steps "[200,300,350,400,450]" -p scheduler_gamma 0.25 \
 -p lambda_ortho 0.5 

# Metatraining
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/meta_basetrain" \
 -p max_train_iter 3000 -p data_folder "data" -p resume "log/${MODEL}/${EXP_ID}/pretrain_basetrain"  \
 -p trainstage metatrain_baseFSCIL -p dataset $DATASET -p average_support_vector_inference True \
 -p random_seed $RANDOM_SEED -p batch_size_training 2 -p batch_size_inference 128 \
 -p optimizer SGD -p sharpening_activation relu -p SGDnesterov True -p representation tanh \
 -p dim_features $DIMFEATURES -p num_ways $WAYS -p num_shots 5 -p block_architecture $MODEL \
 -p learning_rate 0.0005 -p SGDweight_decay 0.00000 -p dropout true -p dropout_rate 0 -p metatrain_frozen False \
 -p metatrain_loss MultiMarginLoss -p loss_param 0.05 -p save_confusion $SAVE_CONFUSION \
 -p scheduler_type CosineAnnealing -p scheduler_warmup_step 100 -p scheduler_min_lr_scaler 0.01

# Evaluation Mode 1 (num_shots relates only to number of shots in base session, on novel there are always 5)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/eval/model"  -p data_folder "data"  \
 -p resume "log/${MODEL}/${EXP_ID}/meta_basetrain" -p dim_features $DIMFEATURES -p retrain_iter 0 -p nudging_iter 0 \
 -p bipolarize_prototypes False -p nudging_act_exp 4 -p nudging_act doubleexp -p trainstage train_FSCIL \
 -p dataset $DATASET -p random_seed $RANDOM_SEED -p learning_rate 0.5 -p batch_size_training 128 -p batch_size_inference 128 \
 -p num_query_training 0 -p optimizer SGD -p sharpening_activation abs -p SGDnesterov True -p representation tanh \
 -p retrain_act tanh -p num_ways $WAYS -p num_shots 100 -p block_architecture $MODEL -p save_confusion $SAVE_CONFUSION \
 -p use_etf_prototypes False 