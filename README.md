# RCRank

This is the official PyTorch code for RCRank.

We propose RCRank, the first method to utilize a multimodal approach for identifying and ranking the root causes of slow queries by estimating their impact. We employ a pre-training method to align the multimodal information of queries, enhancing the performance and training speed of root cause impact estimated. Based on the aligned pre-trained embedding module, we use Cross-modal fusion of feature modalities to ultimately estimate the impact of root causes, identifying and ranking the root causes of slow queries.

## Installation
First clone the repository using Git.

Some data can be downloaded from this [link](https://drive.google.com/file/d/1L26JZDH6TJdleJkGaPbWjNSQm9E3CuO2/view?usp=drive_link), Place the data files into the `data` folder. 

The project dependencies can be installed by executing the following commands in the root of the repository:
```bash
conda env create --name RCRank python=3.9
conda activate RCRank
pip install -r requirements.txt
```

## Pre-training
```bash
python model/pretrain/pretrain.py
```

## Training and Inference
```bash
python train.py
```
