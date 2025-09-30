# Vanilla
# CUDA_VISIBLE_DEVICES=1 python gcn_vanilla.py --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed_num 5

# FairGNN
# CUDA_VISIBLE_DEVICES=1 python gcn_fairgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset bail --seed_num 5

# CUDA_VISIBLE_DEVICES=1 python gcn_fairgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset credit --seed_num 5

# CUDA_VISIBLE_DEVICES=1 python gcn_fairgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset pokec_z --seed_num 5

# CUDA_VISIBLE_DEVICES=1 python gcn_fairgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset pokec_n --seed_num 5

CUDA_VISIBLE_DEVICES=1 python gcn_fairgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset nba --seed_num 5