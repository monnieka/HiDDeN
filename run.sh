#!/bin/bash
#SBATCH --partition=students-dev
#SBATCH --gres=gpu:1
#SBATCH --output output.txt
#SBATCH --error error.txt

python main.py new --name hidden0.0 --data-dir /nas/softechict-nas-2/datasets/coco --batch-size 32
