# Negative to Positive Co-learning with Aggressive Modality Dropout
## Background
This repo contains the code used in the research project [Negative Co-learning to Positive Co-learning with Aggressive Modality Drop](https://drive.google.com/file/d/1bwqcazWJhACQkEVYfpYC_pG_IeetzBvR/view). 
We find that by using aggressive modality
dropout we are able to reverse negative
co-learning (NCL) to positive co-learning
(PCL). Aggressive modality dropout can
be used to ’prep’ a multimodal model for
unimodal deployment, and dramatically increases model performance during negative co-learning, where during some experiments we saw a 20% gain in accuracy.

## Data
You will need both the MOSI dataset and the IEMOCAP dataset to run our experiments. The MOSI dataset with features extracted can be obtained from [here](https://github.com/pliang279/MFN/tree/master/data), while the raw features for IEMOCAP can be downloaded [here](https://sail.usc.edu/iemocap/) after requesting access. To extract and align the different modalities, please follow the steps outlined in our paper.

## File Structure

* `code:`
  * `analysis.ipynb:` File used to create confusion matrix comparing results using modality dropout and results not using modality dropout.
  * `data_stacker.py:` Given that we have already extracted the features from IEMOCAP, this contains the code used to transform our dataset into the format used by our models. Also, this script drops the data points associated with 'xxx', 'other', and missing data.
  * `models.py:` This contains the code for the biEFLSTM and the Memory Fusion Network.
  * `run_model_IEMOCAP.ipynb:` This contains the code used to test modality dropout during negative co-learning on the IEMOCAP dataset.
  * `run_model_MOSI.ipynb:` This contains the code used to test modality dropout during positive co-learning on the MOSI dataset.
  * `training_loops.py:` This contains the code containing the training and validation loops for training and validating models.
* `results:` This directory contains the output results of our experiments.
