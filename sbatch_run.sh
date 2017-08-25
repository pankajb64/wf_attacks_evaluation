#!/bin/bash
#
#SBATCH --output=logs/res_%j.txt
#SBATCH -e logs/res_%j.err
#SBATCH --partition titanx-long
#SBATCH --gres gpu:4
#SBATCH --mem=50000

#python run_exp.py
#echo $1
python run_exp_mta3b.py $1
