import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch_geometric.data import Data as gData
from torch_geometric.loader import DataLoader as gDataLoader

from make_network import *
from utils import *
from modules import FCNN

import argparse

parser = argparse.ArgumentParser()

################################################################
#                       DATA ARGUMENTS                         #
################################################################
parser.add_argument("--data_dir",  type=str, default=None)
parser.add_argument("--res_dir",   type=str, default=None)

parser.add_argument("--seed",      type=int, default=None)

# SPLIT
parser.add_argument("--train_size",  type=float, default=0.6)
parser.add_argument("--valid_size",  type=float, default=0.2)
parser.add_argument("--test_size",   type=float, default=0.2)

################################################################
#                     TRAINING ARGUMENTS                       #
################################################################
parser.add_argument("--epochs",     type=int, default=100)
parser.add_argument("--batch_size",   type=int, default=32)

################################################################
#                       MODEL ARGUMENTS                        #
################################################################
# params
parser.add_argument("--lr",             type=float, default=1e-3)
parser.add_argument("--weight_decay",   type=float, default=1e-3)
parser.add_argument("--dropout",        type=float, default=0.0)
parser.add_argument("--bn",             type=bool,  default=False)

# architecture
parser.add_argument("--fcnn_num_layers",  type=int,  default=2)
parser.add_argument("--fcnn_hid_dim",     type=int,  default=50)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

exp_folder = "FCNN"
if not os.path.exists(args.res_dir + exp_folder):
    os.makedirs(args.res_dir + exp_folder)

# 1 read data
df = pd.read_csv(args.data_dir + "/vols.csv")
vcols = pd.read_csv(args.data_dir + "/rois.csv")["roi"].values.tolist()
num_nodes = len(vcols)

print(df.shape, num_nodes)

# 2 preprocessing - normalize by intracranial volume
for c in vcols:
    df[c] = df[c] / df["intra_vol"]

# 3 split data
idx_train, idx_test  = train_test_split(np.arange(df.shape[0]), test_size=args.test_size,        random_state=args.seed)
idx_train, idx_valid = train_test_split(idx_train, test_size=args.valid_size/(1-args.test_size), random_state=args.seed)

# 4 scale (standardize) data
mean, std = df.iloc[idx_train][vcols].mean(), df.iloc[idx_train][vcols].std()
df[vcols] = (df[vcols] - mean) / std

# 5 create data lists
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dls = list()
for i in range(df.shape[0]):
    d = gData()
    d['x']   = torch.FloatTensor( df.iloc[i][vcols].values ).view(1,-1)
    d['y']   = torch.FloatTensor( df.iloc[i]["age"].ravel() )
    d.to(device)
    dls.append(d)
    
# 6 create data loaders
train_loader = gDataLoader([dls[i] for i in idx_train], batch_size=args.batch_size)
valid_loader = gDataLoader([dls[i] for i in idx_valid], batch_size=args.batch_size)
test_loader  = gDataLoader([dls[i] for i in idx_test],  batch_size=args.batch_size)

# 7 instantiate model
seed_everything(args.seed)
model = FCNN(in_dim=num_nodes,
            hid_dim=args.fcnn_hid_dim, num_layers=args.fcnn_num_layers-1,
            dropout=args.dropout, bn=args.bn).to(device)

# 8 train model
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

metrics = train_and_eval(args.epochs, model, 
                        train_loader, valid_loader,
                        criterion, optimizer, device,
                        score="-mae", best_model_path= args.res_dir + exp_folder + "/s{}.pt".format(args.seed))

# 9 save figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(20,10))
for i, m in enumerate(['mse','mae','corr']):
    ax[i].plot([rep['train'][m] for rep in metrics])
    ax[i].plot([rep['valid'][m] for rep in metrics])
plt.savefig(args.res_dir + exp_folder + "/s{}_fig.png".format(args.seed))

# 10 test best checkpoint
bestmodel = FCNN(in_dim=num_nodes,
            hid_dim=args.fcnn_hid_dim, num_layers=args.fcnn_num_layers-1,
            dropout=args.dropout, bn=args.bn).to(device)
ckpt = torch.load(args.res_dir + exp_folder + "/s{}.pt".format(args.seed))
bestmodel.load_state_dict( ckpt["model_state_dict"] )

metrics = dict()
metrics["train"] = eval_epoch(bestmodel, train_loader, criterion, device) 
metrics["valid"] = eval_epoch(bestmodel, valid_loader, criterion, device) 
metrics["test"]  = eval_epoch(bestmodel, test_loader, criterion, device)
pd.DataFrame(metrics).to_csv( args.res_dir + exp_folder + "/s{}_metrics.csv".format(args.seed))