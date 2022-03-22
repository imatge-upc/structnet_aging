import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, bn=False):
        super(LinearBlock, self).__init__()

        self.lin  = nn.Linear(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim) if bn else nn.Identity()
        self.do   = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.act  = nn.ReLU()
    
    def forward(self, x):
        return self.act( self.do( self.bn( self.lin(x) ) ) )

class PermutInvGP(nn.Module):
    def __init__(self):
        super(PermutInvGP, self).__init__()

    def forward(self, x, batch):
        x = torch.cat([gnn.global_max_pool(x, batch), gnn.global_add_pool(x, batch)], dim=1)
        return x

class ROIAwareGP(nn.Module):
    def __init__(self, num_nodes, num_heads=1):
        super(ROIAwareGP, self).__init__()

        self.num_nodes = num_nodes
        self.num_heads = num_heads

        w = torch.rand( (num_nodes, num_heads) )
        self.w   = nn.parameter.Parameter(w, requires_grad=True)

        self.softmax = nn.Softmax(0)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, batch):
        x, _ = to_dense_batch(x, batch)
        x = (x.transpose(1,2) @ self.softmax(self.w)).transpose(1,2)
        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        return x
    
class PermutEquivMP(nn.Module):
    def __init__(self, in_dim, 
                 hid_dim=128, num_layers=2, 
                 dropout=0.0, bn=False):
        super(PermutEquivMP, self).__init__()

        gls = list()
        gls.append( self._create_conv_layer(in_dim, hid_dim, hid_dim, 
                                            dropout=0.0, bn=False) )
        for l in range(num_layers - 1):
            gls.append( self._create_conv_layer(hid_dim, hid_dim, hid_dim,
                                                dropout=0.0, bn=False) )
        
        self.gls = nn.ModuleList(gls)
    
    def _create_conv_layer(self, in_dim, hid_dim, out_dim, 
                                 bn=False, dropout=0.0):
        local_nn = nn.Sequential(
            LinearBlock(in_dim, hid_dim, dropout=dropout, bn=bn),
            nn.Linear(hid_dim, out_dim)
        )
        gconv = gnn.GINConv(nn=local_nn)
        return gconv
    
    def forward(self, x, edge_index):
        for l in self.gls:
            x = l(x, edge_index)
        return x

class ROIAwareMP(nn.Module):
    def __init__(self, in_dim, num_nodes, 
                 hid_dim=128, num_layers=2, 
                 dropout=0.0, bn=False):
        super(ROIAwareMP, self).__init__()

        gls = list()
        gls.append( self._create_conv_layer(in_dim, num_nodes, hid_dim, hid_dim, 
                                            dropout=0.0, bn=False) )
        for l in range(num_layers - 1):
            gls.append( self._create_conv_layer(hid_dim, num_nodes, hid_dim, hid_dim, 
                                                dropout=0.0, bn=False) )
        
        self.gls = nn.ModuleList(gls)

    def _create_conv_layer(self, in_dim, num_nodes, hid_dim, out_dim, 
                                bn=False, dropout=0.0):
        local_nn = nn.Sequential(
            LinearBlock(in_dim + num_nodes, hid_dim, dropout=dropout, bn=bn),
            nn.Linear(hid_dim, out_dim)
        )
        global_nn = nn.Sequential(
            LinearBlock(hid_dim, hid_dim, dropout=dropout, bn=bn),
            nn.Linear(hid_dim, out_dim)
        )
        gconv = gnn.PointNetConv(local_nn, global_nn)
        return gconv

    def forward(self, x, pos, edge_index):
        for l in self.gls:
            x = l(x, pos, edge_index)
        return x

    
class GNN(nn.Module):
    def __init__(self, in_dim, num_nodes, 
                        msgp="RA", gpool="RA",
                        gnn_hid_dim=128, gnn_num_layers=2,
                        dropout=0.0, bn=False):
        super(GNN, self).__init__()
        assert msgp  in ["RA","PE"], "Message Passing must be either RA (ROI-aware) or PE (Permutation Equivariant)"
        assert gpool in ["RA","PI"], "Global Pooling must be either RA (ROI-aware) or PI (Permutation Invariant)"

        self.msgp  = msgp
        self.gpool = gpool

        if msgp == "RA":
            self.msgp = ROIAwareMP(in_dim, num_nodes, 
                                hid_dim=gnn_hid_dim, num_layers=gnn_num_layers,
                                dropout=dropout, bn=bn)
        elif msgp == "PE":
            self.msgp = PermutEquivMP(in_dim,
                                hid_dim=gnn_hid_dim, num_layers=gnn_num_layers,
                                dropout=dropout, bn=bn)
        
        if gpool == "RA":
            self.gpool = ROIAwareGP(num_nodes, num_heads=2)
        elif gpool == "PI":
            self.gpool = PermutInvGP()
        
        self.fcnn = nn.Sequential(
            LinearBlock(2 * gnn_hid_dim, gnn_hid_dim, dropout=dropout, bn=bn),
            nn.Linear(gnn_hid_dim, 1)
        )
    
    def forward(self, data):
        # message-passing
        if self.msgp == "RA":
            x = self.msgp(data.x, data.pos, data.edge_index)
        else:
            x = self.msgp(data.x, data.edge_index)

        # global pooling
        x = self.gpool(x, data.batch)

        # FCNN regressor
        x = self.fcnn(x)

        return x

class FCNN(nn.Module):
    def __init__(self, in_dim, 
                 hid_dim=128, num_layers=2, 
                 dropout=0.0, bn=False):
        super(FCNN, self).__init__()
        
        mls = list()
        mls.append(LinearBlock(in_dim, hid_dim, 
                               dropout=dropout, bn=bn))
        for l in range(num_layers-1):
            mls.append(LinearBlock(hid_dim, hid_dim, 
                                   dropout=dropout, bn=bn))
        
        mls.append(nn.Linear(hid_dim, 1))
        self.mls = nn.Sequential(*mls)

        print(self)
        
    def forward(self, data):
        return self.mls(data.x)  