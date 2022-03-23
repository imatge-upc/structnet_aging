# Structural Networks for Brain Age Prediction
Official repository of the paper [Structural Networks for Brain Age Prediction](https://openreview.net/forum?id=Uf8Ow26cpU&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DMIDL.io%2F2022%2FConference%2FAuthors%23your-submissions)), accepted to MIDL 2022 Conference. 

## Abstract
Biological networks have gained considerable attention within the Deep Learning community because of the promising framework of Graph Neural Networks (GNN), neural models that operate in complex networks. In the context of neuroimaging, GNNs have successfully been employed for functional MRI processing but their application to ROI-level structural MRI (sMRI) remains mostly unexplored.
In this work we analyze the implementation of these geometric models with sMRI by building graphs of ROIs (ROI graphs) using tools from Graph Signal Processing literature and evaluate their performance in a downstream supervised task, age prediction. We first make a qualitative and quantitative comparison of the resulting networks obtained with common graph topology learning strategies. In a second stage, we train GNN-based models for brain age prediction. Since the order of every ROI graph is exactly the same and each vertex is an entity by itself (a ROI), we evaluate whether including ROI information during message-passing or global pooling operations is beneficial and compare the performance of GNNs against a Fully-Connected Neural Network baseline.
The results show that ROI-level information is needed during the global pooling operation in order to achieve competitive results. However, no relevant improvement has been detected when it is incorporated during the message passing. These models achieve a MAE of 4.27 in hold-out test data, which is a performance very similar to the baseline, suggesting that the inductive bias included with the obtained graph connectivity is relevant and useful to reduce the dimensionality of the problem.

## Code

Models are implemented in PyTorch and PyTorch Geometric.

- *utils.py*: contains functions needed to read the data, train and evaluate the models.
- *make_network.py*: functions to create association networks given a set of observations (numpy array). 
- *modules.py*: modules and models that has been employed.
- *train_fcnn.py*: script to train and evaluate a FCNN model, for an example of how to run the script, see run_fcnn.py.
- *run_fcnn.py*: run train_fcnn.py with given arguments.
- *train_gnn.py*: script to train and evaluate a GNN model, for an example of how to run the script, see run_gnn.py
- *run_gnn.py*: run train_gnn.py with given arguments.
- *network_topology_inference.ipynb*: evaluation of the network topology inference strategies and the effect the sparsity hyperparameter has on the obtained networks. Finally, the approach of group structural differences is exemplified by comparing the networks of APOE carriers vs non-carriers.
