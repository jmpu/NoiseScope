#!/usr/bin/env bash

REAL_RES_DIR='/rdata/jiameng/DeepLens/alldata/residuals/flickr_bulldog/real/'
FAKE_RES_DIR='/rdata/jiameng/DeepLens/alldata/residuals/bulldog_.86/fake/'
REFER_RES_DIR='/rdata/jiameng/DeepLens/alldata/residuals/flickr_bulldog/refer/'
NUM_REAL=500
NUM_FAKE=500
IMG_DIM=256
OUTLIER_MODEL_PATH='./fingerprint_classifier/flickr_bulldog_outlier_model.pkl'
RESULT_DIR='./detection_result/'
PCE_THRE='9.77'

python noisescope_clustering.py \
--real_res_dir=${REAL_RES_DIR} \
--fake_res_dir=${FAKE_RES_DIR} \
--refer_res_dir=${REFER_RES_DIR} \
--num_real=${NUM_REAL} \
--num_fake=${NUM_FAKE} \
--img_dim=${IMG_DIM} \
--outlier_model_path=${OUTLIER_MODEL_PATH} \
--result_dir=${RESULT_DIR} \
--pce_thre=${PCE_THRE}


# Estimated T-merge used in the paper:
# T_merge for dataset BigGAN_DogHV: 9.95
# T_merge for dataset CycleGAN_winter: 11.68
# T_merge for dataset PGGAN_tower: 26.97
# T_merge for both StyleGAN_face1 and StyleGAN_face2: 8.45

# StyleGAN_face1 and StyleGAN_face2 share the same real image dataset and reference image dataset. Thus they share
# the same T_merge value and the same pretrained fingerprint classifier.

# For StyleGAN_face1 and StyleGAN_face2: parameter img_dim in prep_steps.py and run_NoiseScope.sh should be set as 1024.
# For BigGAN_DogHV, CycleGAN_winter and PGGAN_tower, parameter img_dim in prep_steps.py and run_NoiseScope.sh should be set as 256.
# If the image set you test on has images of different sizes, Please resize all the images to squares with the same length before running any script.