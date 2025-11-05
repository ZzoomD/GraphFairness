#!/bin/bash

# FairDLA运行脚本，基于FairSAD项目中的最佳参数设置

echo "Running FairDLA with best hyper-parameters from FairSAD"

# German数据集的最佳参数
echo '============German============='
python gcn_fairdla.py --dataset german --epochs 1000 --seed_num 5 --lr 1e-2 --nhid 16 --dropout 0.5 --alpha 0.1 --channels 2 --pre_train 0 --device 0 --avgy False --rs 10 --per 0.3 --adv 0

# Bail数据集的最佳参数
echo '============Bail============='
python gcn_fairdla.py --dataset bail --epochs 1000 --seed_num 5 --lr 1e-3 --nhid 16 --dropout 0.5 --alpha 0.001 --channels 2 --pre_train 0 --device 0 --avgy False --rs 10 --per 0.3 --adv 0

# Credit数据集的最佳参数
echo '============Credit============='
python gcn_fairdla.py --dataset credit --epochs 1000 --seed_num 5 --lr 1e-3 --nhid 16 --dropout 0.5 --alpha 0.5 --channels 2 --pre_train 0 --device 0 --avgy False --rs 10 --per 0.3 --adv 0

# Pokec_z数据集的最佳参数
echo '============Pokec_z============='
python gcn_fairdla.py --dataset pokec_z --epochs 1000 --seed_num 5 --lr 1e-3 --nhid 16 --dropout 0.5 --alpha 0.001 --channels 2 --pre_train 0 --device 0 --avgy False --rs 10 --per 0.3 --adv 0

# Pokec_n数据集的最佳参数
echo '============Pokec_n============='
python gcn_fairdla.py --dataset pokec_n --epochs 1000 --seed_num 5 --lr 1e-3 --nhid 16 --dropout 0.5 --alpha 0.05 --channels 2 --pre_train 0 --device 0 --avgy False --rs 10 --per 0.3 --adv 0

echo "All experiments completed!"