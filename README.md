# combgym


## Overview

This project utilizes several models for analysis and prediction, including Ridge, CNN, ESM-1b, ESM-1v, BlOSUM62, GVP-Mut, MAVE-NN, DeepSequence, and EVmutation. Below are the sources and relevant links for each model:

## Models and Sources

- **Ridge, CNN, ESM-1b, ESM-1v, BlOSUM62**
  - Source: FLIP by Christian Dallago et al.
  - GitHub Link: [FLIP](https://github.com/J-SNACKKB/FLIP)
  - Provided Scripts:
    - `mutant to seq.ipynb`: Generates mutants from sequences.
    - `seq to mutant.ipynb`: Generates sequences from mutants.
    - `split.ipynb`: Splits the data into training and testing sets.
    - `dictionary.ipynb`: Generates a dictionary to be added to the `train_all.py` file.
    - `run_train_all.sh`: Executes predictions on files within a directory in batch mode.
    - `statistics.py`: Evaluates Spearman and NDCG metrics in batch mode.


- **GVP-Mut**
  - Source: GVP-MSA by Lin Chen et al.
  - GitHub Link: [GVP-MSA](https://github.com/cl666666/GVP-MSA)
  - Provided Script:
   - `gvpmutsplit.ipynb`: Splits data using a strategy based on `train_single2multi.py` from GVP-MSA, but without using  MSA.


- **MAVE-NN**
  - Source: MAVE-NN by Ammar Tareen et al.
  - GitHub Link: [MAVE-NN](https://github.com/jbkinney/mavenn)
  - Provided Scripts:
    - `mavenn_prediction.ipynb`: Used for prediction and visualization on a single dataset.
    - `run_mavenn.py`: Used for batch prediction on multiple files.


- **DeepSequence**
  - Source: DeepSequence by Thomas A Hopf et al.
  - GitHub Link: [DeepSequence](https://github.com/debbiemarkslab/DeepSequence)
  - Provided Script:
    - `run_predict.py`: Used for making predictions.


- **EVmutation**
  - Source: EVmutation by Thomas A Hopf et al.
  - GitHub Link: [EVmutation](https://github.com/debbiemarkslab/EVmutation)
  - Provided Script:
    - `multi_prediction.ipynb`: Used for making predictions on multipoint datasets.


## Installation and Usage

Please refer to the instructions on the respective GitHub pages of each model for installation and usage.


