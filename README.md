# rgcn_paddlepaddle
Modeling Relational Data with Graph Convolutional Networks in PaddlePaddle

This is a PaddlePaddle implementation of the Relational Graph Convolutional Networks (R-GCN) described in the paper:

Schlichtkrull, Michael, et al. "Modeling relational data with graph convolutional networks." European semantic web conference. Springer, Cham, 2018.

The code in this repo is based on or refers to https://github.com/berlincho/RGCN-pytorch and https://github.com/tkipf/relational-gcn

# Requirements
* Hardware：CPU (RAM larger than 36G is recommended)
* python-3.8.12
* paddlepaddle-2.1.3
* paddlenlp-2.1.1
* rdflib-6.0.2
* wget-3.2
* h5py-3.5.0
* install requirements via pip install -r requirements.txt

# Usage
train: python run.py --train

test: python run.py
