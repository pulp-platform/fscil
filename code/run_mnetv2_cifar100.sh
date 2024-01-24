#!/bin/bash


EXP_ID="001"
SAVE_CONFUSION=True
CUDA_DEVICES='0'
MODEL=mnetv2_x4
# MODEL=mnetv2_x2
# MODEL=mnetv2_x4
# MODEL=mini_resnet12

DIMFEATURES=256
WIDTH_MULT=1
ROUND_NEAREST=1
WAYS=60
BATCH_TRAIN=10
RANDOM_SEED=8

# Pretraining
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/pretrain_basetrain" \
 -p max_train_iter 200 -p data_folder "../data" -p trainstage pretrain_baseFSCIL -p dataset cifar100 \
 -p random_seed $RANDOM_SEED -p learning_rate 0.025 -p batch_size 128 -p optimizer SGD \
 -p SGDnesterov True -p representation real -p dim_features $DIMFEATURES -p block_architecture $MODEL \
 -p SGDweight_decay 0.0005 -p SGDmomentum 0.9 -p advance_augment True -p dropout true -p dropout_rate 0.0 \
 -p lambda_ortho 0.5

# Metatraining
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/meta_basetrain" \
 -p max_train_iter 3000 -p data_folder "../data" -p resume "log/${MODEL}/${EXP_ID}/pretrain_basetrain"  \
 -p trainstage metatrain_baseFSCIL -p dataset cifar100 -p average_support_vector_inference True \
 -p random_seed $RANDOM_SEED -p batch_size_training 10 -p batch_size_inference 128 \
 -p optimizer SGD -p sharpening_activation relu -p SGDnesterov True -p representation real \
 -p dim_features $DIMFEATURES -p num_ways $WAYS -p num_shots 20 -p block_architecture $MODEL \
 -p learning_rate 0.1 -p SGDweight_decay 0.000000 -p dropout true -p dropout_rate 0 -p metatrain_frozen False \
 -p metatrain_loss MultiMarginLoss -p loss_param 0.1 -p save_confusion $SAVE_CONFUSION \
 -p scheduler_type CosineAnnealing -p scheduler_warmup_step 100 -p scheduler_min_lr_scaler 0.01 -p normalize_weightings False

# Evaluation Mode 1 (num_shots relates only to number of shots in base session, on novel there are always 5)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/eval/model"  -p data_folder "../data"  \
 -p resume "log/${MODEL}/${EXP_ID}/meta_basetrain" -p dim_features $DIMFEATURES -p retrain_iter 0 -p nudging_iter 0 \
 -p bipolarize_prototypes False -p nudging_act_exp 4 -p nudging_act doubleexp -p trainstage train_FSCIL \
 -p dataset cifar100 -p random_seed $RANDOM_SEED -p learning_rate 0.0025 -p batch_size_training 128 -p batch_size_inference 128 \
 -p num_query_training 0 -p optimizer SGD -p sharpening_activation relu -p SGDnesterov True -p representation real \
 -p retrain_act tanh -p num_ways $WAYS -p num_shots 200 -p block_architecture $MODEL -p save_confusion $SAVE_CONFUSION \
 -p use_etf_prototypes False -p SGDweight_decay 0.0000 -p lr_step_size 300

# # Evaluation Mode 2 (num_shots relates only to number of shots in base session, on novel there are always 5)
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u main.py simulation -v -ld "log/${MODEL}/${EXP_ID}/eval/model"  -p data_folder "../data"  \
#  -p resume "log/${MODEL}/${EXP_ID}/meta_basetrain" -p dim_features $DIMFEATURES -p retrain_iter 1000 -p nudging_iter 0 \
#  -p bipolarize_prototypes True -p nudging_act_exp 4 -p nudging_act doubleexp -p trainstage train_FSCIL \
#  -p dataset cifar100 -p random_seed $RANDOM_SEED -p learning_rate 0.0025 -p batch_size_training 128 -p batch_size_inference 128 \
#  -p num_query_training 0 -p optimizer SGD -p sharpening_activation abs -p SGDnesterov True -p representation tanh \
#  -p retrain_act tanh -p num_ways $WAYS -p num_shots 200 -p block_architecture $MODEL -p save_confusion $SAVE_CONFUSION \
#  -p use_etf_prototypes False -p SGDweight_decay 0.0000 -p lr_step_size 300
