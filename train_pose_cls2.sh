#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python example/mpii.py \
 -a hg \
 --stacks 2 \
 --blocks 1 \
 --checkpoint checkpoint/mpii/hg2_cls_8 \
 -j 4 \
 --sigma 0.55 \
 --train-batch 34 \
 --resume checkpoint/mpii/hg2_cls_8/checkpoint.pth.tar
