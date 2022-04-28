#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J fl_bd
#SBATCH -p datasci
#SBATCH --output=result/WED20042022/RLR_100clients20compromised.out
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
module load python
conda activate dgl
python -u ./fl_RLR_attack.py --use_org_node_attr --train_verbose --target_class 0 --train_epochs 5 --fl_epochs 50 --num_client 100 --num_compromised_client 20 --rlr_thres 0.01
