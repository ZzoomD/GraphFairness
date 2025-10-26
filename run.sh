# Vanilla
# CUDA_VISIBLE_DEVICES=1 python gcn_vanilla.py --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed_num 5

# FairGNN
# CUDA_VISIBLE_DEVICES=1 python gcn_fairgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed_num 5

# FairVGNN
# CUDA_VISIBLE_DEVICES=1 python gcn_fairvgnn.py --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed_num 5

# FairDrop
CUDA_VISIBLE_DEVICES=1 python gcn_fairdrop.py --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed_num 5 --delta 0.25
