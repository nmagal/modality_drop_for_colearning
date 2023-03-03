# Negative to Positive Co-learning with Aggressive Modality Dropout

This repo contains the code used in the research project [Negative Co-learning to Positive Co-learning with Aggresive Modality Drop](https://drive.google.com/file/d/1bwqcazWJhACQkEVYfpYC_pG_IeetzBvR/view). 
We find that by using aggressive modality
dropout we are able to reverse negative
co-learning (NCL) to positive co-learning
(PCL). Aggressive modality dropout can
be used to ’prep’ a multimodal model for
unimodal deployment, and dramatically increases model performance during negative co-learning, where during some experiments we saw a 20% gain in accuracy.

## Running 
In order to run the code for our experiments using a bi-EFLSTM, please run run_model_IEMOCAP.ipynb. To run our expierements for the Memory Fusion Network (MFN), please run run_model_MOSI.ipynb.
