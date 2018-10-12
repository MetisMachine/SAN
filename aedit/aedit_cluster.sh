#!/usr/bin/env sh
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices and detector and the number of cluster"
  exit 1
fi
gpus=$1
model=vgg16_base
cluster=$3
batch_size=8
height=224
width=224
dataset_name=aedit_2

CUDA_VISIBLE_DEVICES=${gpus} python cluster.py \
    --style_train_root ~/datasets/aedit-Style/ \
    --train_list ~/datasets/aedit-Style/aedit-training.txt ~/datasets/aedit-Style/aedit-testing.txt \
    --learning_rate 0.01 --epochs 2 \
    --save_path ./snapshots/CLUSTER-${dataset_name}-${cluster} \
    --num_pts 78 --pre_crop_expand 0.2 \
    --arch ${model} --cpm_stage 3 \
    --dataset_name ${dataset_name} \
    --scale_min 1 --scale_max 1 --scale_eval 1 --eval_batch ${batch_size} --batch_size ${batch_size} \
    --crop_height ${height} --crop_width ${width} --crop_perturb_max 30 \
    --sigma 3 --print_freq 5 --print_freq_eval 10 --pretrain \
    --evaluation --heatmap_type gaussian --argmax_size 3 --n_clusters ${cluster}
