# Application of HyenaDNA for RNA Splicing Prediction
This repository contains the final project of the NLP course completed by Ella Rannon, Gal Bodek and Keren Danan.
In this project, we investigated the use of HyenaDNA, a novel genomic foundational model, for exon prediction, exploring various input lengths and model
architectures.

This repository contains multiple files: 

**Data:**

1- **create_encoded_dict.py** - Python file encoding transcript sequences to 1's and 0's encoding indicating exons/introns nucleotides.

2- **create_data_script_loaded.py** - Python file generating data from encoded sequences as described in our methods. 

3- **prepare_dataset.py** - Python file loading the data json file generated, splitting it to train and validation sets, and then saving them in smaller json files for the model input.

4- **create_dataset.py** - Python file for the creation of the dataset object used for classification during the run of the main model.

5- **analysis_script.py** - Python file generating statistics about the data. 

**Model:**

1- **HyenaDNA.py** - The model itself, as published by the original authors of HyenaDNA paper.

2- **classification_model.py** - Class of the classification model used in this project, which uses a pre-trained HyenaDNA model and additional dense layers.

3- **train_model.py** - Train the model and evaluate it on the validation set.

4- **evalutate_model.py** - Run the trained model on a validation \ test set and reporting the results.

5- **utilities.py** - File for utility functions, such as calculating training loss and other performance metrics, and preparing data batch.

6- **checkpoints directory** - Directory with the configuration files for the HyenaDNA models.

**Usage:**

In order to train the model, you should use the following command:
python <path_to_code_dir>/train_model.py --data_file <path_to_data_dir> --save-prefix <model_output_prefix> --dropout <dropout> -e <epoch_num> --batch_size <batch_size> --device <device> --num_layers <layer_num> -ac <batch_num_to_accumulate> --lr <learning_rate> --save-interval <save_interval> --weight_decay <weight_decay>


For further information, you can contact us by mail: ellarannon@mail.tau.ac.il, galbodek@mail.tau.ac.il, kerendanan1@mail.tau.ac.il
