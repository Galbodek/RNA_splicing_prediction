#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -r y
#PBS -q gpu2
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N train_splicing
#PBS -o /sternadi/nobackup/volume1/ellarannon/splicing
#PBS -e /sternadi/nobackup/volume1/ellarannon/splicing
#PBS -l select=1:mem=100gb:ncpus=2:ngpus=1
####PBS -l select=1:mem=60gb:ncpus=8:ngpus=1:host=compute-0-58

id
date
hostname

source ~/.power_bashrc
conda activate splicing
export PATH=$CONDA_PREFIX/bin:$PATH

python "/sternadi/nobackup/volume1/ellarannon/splicing/train_model.py" --data_file "/sternadi/nobackup/volume1/ellarannon/splicing/data" --dropout 0.1 -e 3 --save-prefix "/sternadi/nobackup/volume1/ellarannon/splicing/hyena_model_" --device 1 --batch_size 2 --h_dim 128
#################python /sternadi/nobackup/volume1/ellarannon/splicing/wow.py