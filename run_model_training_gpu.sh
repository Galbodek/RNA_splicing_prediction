#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -r y
#PBS -q gpu2
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N train_splicing
#PBS -o /davidb/ellarannon/splicing
#PBS -e /davidb/ellarannon/splicing
#PBS -l select=1:mem=150gb:ncpus=2:ngpus=1
####PBS -l select=1:mem=60gb:ncpus=8:ngpus=1:host=compute-0-58

id
date
hostname

source ~/.bashrc
conda activate splicing
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# ###############export CUDA_VISIBLE_DEVICES=4


python "/davidb/ellarannon/splicing/train_model.py" --data_file "/davidb/ellarannon/splicing/updated_all_data_human_600/" --dropout 0.1 -e 3 --save-prefix "/davidb/ellarannon/splicing/600__hyena_model_"  --batch_size 64 --h_dim 128 --device 7 --num_layers 3 -ac 1 --lr 0.000001 --save-interval 10000
