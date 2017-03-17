#!/bin/bash
# template for tensorflow script
# tru <mailto:tru@pasteur.fr>
# https://github.com/DeepLearningClub/pasteur-gpu-cluster
#
#SBATCH --job-name=tensorflow-gpu
#SBATCH --time=2:00
# Acceptable time formats  include
# "minutes",  "minutes:seconds", "hours:minutes:seconds",
# "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=gpu
#SBATCH --gres=gpu:teslaP100:1
# choice of gpu flavor: teslaM40 (max: 4) teslaK80 (max: 8) teslaP100 (max: 16)
#SBATCH --mail-type=ALL
# choice of: NONE, BEGIN, END, FAIL, REQUEUE, ALL

#hostname
#head /proc/cpuinfo
#printenv|grep SLURM

# tf setup
module load cuda/8.0.0 cudnn/v5 tensorflow/1.0.0-py2 Python/2.7.11
# run the python with gpu
srun --gres=gpu:1 python <<EOF
import tensorflow as tf
tf.Session(config=tf.ConfigProto(log_device_placement=True))
EOF
# or if you have a python file command:
#srun --gres=gpu:1 python ~/python/tf-check-gpu.py

