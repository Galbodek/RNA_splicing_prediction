# Application of HyenaDNA for RNA Splicing Prediction
This repository contains the final project of the NLP course completed by Ella Rannon, Gal Bodek and Keren Danan.
In this project, we investigated theuse of HyenaDNA, a novel genomic foundational model, for exon prediction, exploring various input lengths and model
architectures.

This repository contains multiple files: 

**Data:**

1- **create_encoded_dict.py** - Python file encoding transcript sequences to 1's and 0's encoding indicating exons/introns nucleotides.

2- **create_data_script_loaded.py** - Python file generating data from encoded sequences as described in our methods. 

3- **prepare_dataset.py** - Python file loading the data json file generated, and splitting it for the model input.

4- **create_dataset.py** - Python file creating class for the classification of the data during the run of the main model.

5- **analysis_script.py** - Python file generating statistics about the data. 

**Model:**

1- **HyenaDNA.py** - the model itself

2- **model_eval.py** - 

3- **evalutate_model.py** -

4- **utilities.py**

5- **run_model_training_gpu.sh** - bash file creatingn run for training the model on specific-length data on the server GPUs.

6- **checkpoints directory** - 


For further information you can contact us by mail: ellarannon@mail.tau.ac.il, galbodek@mail.tau.ac.il, kerendanan1@mail.tau.ac.il
