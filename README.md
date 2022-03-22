# structnet_aging
Official repository of the paper [Structural Networks for Brain Age Prediction](https://openreview.net/forum?id=Uf8Ow26cpU&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DMIDL.io%2F2022%2FConference%2FAuthors%23your-submissions)), accepted to MIDL 2022 Conference. Models are implemented in PyTorch and PyTorch Geometric.

- *utils.py*: contains functions needed to read the data, train and evaluate the models.
- *make_network.py*: functions to create association networks given a set of observations (numpy array). 
- *modules.py*: modules and models that has been employed.
- *train_fcnn.py*: script to train and evaluate a FCNN model, for an example of how to run the script, see run_fcnn.py.
- *run_fcnn.py*: run train_fcnn.py with given arguments.
- *train_gnn.py*: script to train and evaluate a GNN model, for an example of how to run the script, see run_gnn.py
- *run_gnn.py*: run train_gnn.py with given arguments.
- *network_topology_inference.ipynb*: evaluation of the network topology inference strategies and the effect the sparsity hyperparameter has on the obtained networks. Finally, the approach of group structural differences is exemplified by comparing the networks of APOE carriers vs non-carriers.
