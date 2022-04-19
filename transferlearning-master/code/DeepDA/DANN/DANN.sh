#!/usr/bin/env bash
GPU_ID=1
data_dir=/public/home/dongsx/cv2/hw2/office31
# Office31
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain amazon | tee DANN_D2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain dslr --tgt_domain webcam | tee DANN_D2W.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain dslr | tee DANN_A2D.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain amazon --tgt_domain webcam | tee DANN_A2W.log

#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain amazon | tee DANN_W2A.log
#CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DANN/DANN.yaml --data_dir $data_dir --src_domain webcam --tgt_domain dslr | tee DANN_W2D.log
