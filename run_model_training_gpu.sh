#!/bin/bash
#PBS -S /bin/bash
#PBS -j oe
#PBS -r y
#PBS -q gpu2
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N train_splicing
#PBS -o /davidb/ellarannon/splicing
#PBS -e /davidb/ellarannon/splicing
#PBS -l select=1:mem=75gb:ncpus=2:ngpus=1

id
date
hostname

source ~/.bashrc
conda activate splicing
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python "/davidb/ellarannon/splicing/train_model.py" --data_file "/davidb/ellarannon/splicing/updated_all_data_human_600/" --save-prefix "/davidb/ellarannon/splicing/600__hyena_model_" --dropout 0.1 -e 10 --batch_size 256 --h_dim 128 --device 0 --num_layers 0 -ac 1 --lr 1e-4 --save-interval 10000 --weight_decay 0.0
