# Negative to Positive Co-learning with Aggressive Modality Dropout

This repo contains the code used in the research project [Negative Co-learning to Positive Co-learning with Aggresive Modality Drop](https://drive.google.com/file/d/1bwqcazWJhACQkEVYfpYC_pG_IeetzBvR/view). 
We find that by using aggressive modality
dropout we are able to reverse negative
co-learning (NCL) to positive co-learning
(PCL). Aggressive modality dropout can
be used to ’prep’ a multimodal model for
unimodal deployment, and dramatically increases model performance during negative co-learning, where during some experiments we saw a 20% gain in accuracy.

## Running 
In order to run the code for our experiments using the bi-EFLSTM, please run run_model_IEMOCAP.ipynb. To run our experiments for the Memory Fusion Network (MFN), please run run_model_MOSI.ipynb.

## Data
You will need both the MOSI dataset and the IEMOCAP dataset to run our experiments. The MOSI dataset with features extracted can be obtained from [here] (https://github.com/pliang279/MFN/tree/master/data), while the raw features for IEMOCAP can be downloaded [here] (https://sail.usc.edu/iemocap/) after requesting access. To extract and align the different modalities, please follow the steps outlined in our paper.  
