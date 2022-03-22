import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch_geometric.data import Data as gData
from torch_geometric.loader import DataLoader as gDataLoader

from make_network import *
from utils import *
from modules import GNN

DATA_DIR = "/home/oscar/Documents/PhD/datasets/ukbiobank/fs1/"
F_SUBJ = "ukb_fs_subj.csv"
F_VOL = "ukb_fs_ctx_vol.csv"
F_SVOL = "ukb_fs_sctx_vol.csv"
F_THICK = "ukb_fs_ctx_thick.csv"

DATA_DIR = "./toydata/"

def read_ukb_data(data_dir):
    df_subj  = pd.read_csv(data_dir + F_SUBJ)
    df_vol   = pd.read_csv(data_dir + F_VOL)
    df_svol  = pd.read_csv(data_dir + F_SVOL)
    
    df  = df_subj.merge(df_vol,   how='inner', left_on=['subj_id','instance'],  right_on=['subj_id','instance']) \
                       .merge(df_svol,  how='inner', left_on=['subj_id','instance'],  right_on=['subj_id','instance'])
    df = df[ df["instance"] == 2 ]
    return df

parser = argparse.ArgumentParser()

################################################################
#                       DATA ARGUMENTS                         #
################################################################
parser.add_argument("--data_dir",  type=str, default=DATA_DIR)
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

# 1 read data
df = read_ukb_data(args.data_dir)
vcols = [c for c in df.columns if c.startswith("vol_") and c!="vol_whole_subcortex_subcortgray"]
num_nodes = len(vcols)

# 2 preprocessing
df = df[ df[vcols + ["sex","age","apoe","intra_vol"]].isna().sum(axis=1) == 0 ]
disc_cols = ["trauma_ok","cancer_ok","stroke_ok","icd9_ok","icd10_ok","medication_ok"]
df = df[ df[disc_cols].sum(axis=1) == len(disc_cols) ]
df = df[ (df["ethnic"] == 1001) & (df["instance"] == 2)]

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
            dropout=args.dropout, bn=args.bn)

# 8 train model
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

metrics = train_and_eval(args.num_epochs, model, 
                        train_loader, valid_loader,
                        criterion, optimizer, device,
                        score="-mae", best_model_path="./results/FCNN/s{}.pt".format(args.seed))