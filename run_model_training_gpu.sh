#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -r y
#PBS -q gpu2
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N train_splicing
#PBS -o /sternadi/nobackup/volume1/ellarannon/splicing
#PBS -e /sternadi/nobackup/volume1/ellarannon/splicing
#PBS -l select=1:mem=150gb:ncpus=2:ngpus=1
####PBS -l select=1:mem=60gb:ncpus=8:ngpus=1:host=compute-0-58

id
date
hostname

source ~/.power_bashrc
conda activate splicing
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# ###############export CUDA_VISIBLE_DEVICES=4


python "/sternadi/nobackup/volume1/ellarannon/splicing/train_model.py" --data_file "/sternadi/nobackup/volume1/ellarannon/splicing/data" --dropout 0.1 -e 3 --save-prefix "/sternadi/nobackup/volume1/ellarannon/splicing/hyena_model_"  --batch_size 4 --h_dim 128 #--device 4
