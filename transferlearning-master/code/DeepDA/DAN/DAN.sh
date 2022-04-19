#!/usr/bin/env bash
GPU_ID=0
data_dir=/public/home/dongsx/cv2/hw2/office31
# Office31
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAN/DAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee DAN_D2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAN/DAN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee DAN_D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAN/DAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee DAN_A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAN/DAN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee DAN_A2W.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAN/DAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee DAN_W2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DAN/DAN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee DAN_W2D.log

