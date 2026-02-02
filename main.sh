#!/bin/bash

bash ~/.bashrc
conda activate wham

MODEL=transformer # transformer | stoch_transformer

python main.py --mode train \
--proj_name h2compact --run_name 2025-0505-1325_${MODEL} \
--model ${MODEL} \
--d_model 128 --n_layers 4 --n_heads 4 --drop 0.5 \
--w_kl 1e-3 \
--epoch 4000 --batch 128 \
--lr 1e-3 \
--save