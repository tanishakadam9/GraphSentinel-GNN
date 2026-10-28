from __future__ import annotations
import torch
from torch import nn

class GraphSAGELayer(nn.Module):
    """Mean-neighbor GraphSAGE-style aggregation layer implemented in plain PyTorch."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.self_linear=nn.Linear(in_features,out_features)
        self.neighbor_linear=nn.Linear(in_features,out_features)
        self.norm=nn.LayerNorm(out_features)
    def forward(self,x,adj_norm):
        neighbor_mean=adj_norm @ x
        return torch.relu(self.norm(self.self_linear(x)+self.neighbor_linear(neighbor_mean)))

class GraphSAGEEdgeClassifier(nn.Module):
    """GraphSAGE encoder + binary source/destination event classifier."""
    def __init__(self,node_features:int,edge_features:int,hidden:int=32,dropout:float=0.15):
        super().__init__()
        self.sage1=GraphSAGELayer(node_features,hidden)
        self.sage2=GraphSAGELayer(hidden,hidden)
        self.dropout=nn.Dropout(dropout)
        self.edge_head=nn.Sequential(nn.Linear(hidden*3+edge_features,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def encode(self,x,adj_norm):
        return self.sage2(self.dropout(self.sage1(x,adj_norm)),adj_norm)
    def forward(self,x,adj_norm,src_idx,dst_idx,edge_features):
        z=self.encode(x,adj_norm); zs,zd=z[src_idx],z[dst_idx]
        combined=torch.cat([zs,zd,torch.abs(zs-zd),edge_features],dim=1)
        return self.edge_head(combined).squeeze(1)
