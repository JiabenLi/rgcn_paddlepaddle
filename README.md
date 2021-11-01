# rgcn_paddlepaddle
Modeling Relational Data with Graph Convolutional Networks in PaddlePaddle

This is a PaddlePaddle implementation of the Relational Graph Convolutional Networks (R-GCN) described in the paper:

Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.

The code in this repo is based on or refers to https://github.com/berlincho/RGCN-pytorch

# Requirements
* python-3.8.12
* paddlepaddle-2.1.3
* paddlenlp-2.1.1
* rdflib-6.0.2
* wget-3.2
* h5py-3.5.0
* install requirements via pip install -r requirements.txt

# Trained Model
The paddle model we trained：

Link: https://pan.baidu.com/s/1CyRN0hnzjbOPNi8qMecUMA Password：aadf

# Usage
train: python run.py --train

test: python run.py --train False

# Results
Due to the characteristics of model in the article, the results are fluctuant all the time, so we show the average result: acc=95.83%.
