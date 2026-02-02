#!/bin/bash

bash ~/.bashrc
conda activate wham

MODEL=transformer # transformer | stoch_transformer

WEIGHT=/home/mmvc/Documents/Hao_Huang/projects/WHAM/h2compact/checkpoints/${MODEL}_model_best.pt

python main.py --mode infer \
--model ${MODEL} \
--d_model 128 --n_layers 4 --n_heads 4 --drop 0.5 \
--weight ${WEIGHT}