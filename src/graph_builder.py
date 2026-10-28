from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import torch

def build_graph(df: pd.DataFrame, feature_cols: list[str], all_hosts: list[str] | None = None):
    """Build a directed host graph and normalized tensors from the supplied history only.

    Pass `all_hosts` from the full event table when scoring later events; hosts unseen in
    training are included as isolated nodes, preventing test-edge topology leakage.
    """
    observed = set(df["source_host"].astype(str)) | set(df["destination_host"].astype(str))
    hosts = sorted(set(map(str, all_hosts)) | observed) if all_hosts is not None else sorted(observed)
    host_to_idx = {h:i for i,h in enumerate(hosts)}
    n = len(hosts)
    out_count=np.zeros(n,dtype=np.float32); in_count=np.zeros(n,dtype=np.float32)
    unique_out=[set() for _ in range(n)]; failed=np.zeros(n,dtype=np.float32)
    sums=np.zeros((n,len(feature_cols)),dtype=np.float32); counts=np.zeros(n,dtype=np.float32)
    for _,row in df.iterrows():
        s,d=host_to_idx[str(row.source_host)],host_to_idx[str(row.destination_host)]
        out_count[s]+=1; in_count[d]+=1; unique_out[s].add(d); failed[s]+=float(row.failed_login)
        vals=row[feature_cols].to_numpy(dtype=np.float32)
        sums[s]+=vals; sums[d]+=vals; counts[s]+=1; counts[d]+=1
    means=sums/np.maximum(counts[:,None],1.0)
    node_x=np.column_stack([np.log1p(out_count)/10.0,np.log1p(in_count)/10.0,
        np.array([len(v) for v in unique_out],dtype=np.float32)/max(1,n),np.log1p(failed)/5.0,means]).astype(np.float32)
    adj=np.eye(n,dtype=np.float32)
    graph=nx.DiGraph(); graph.add_nodes_from(hosts)
    for _,row in df.iterrows():
        s,d=host_to_idx[str(row.source_host)],host_to_idx[str(row.destination_host)]
        adj[s,d]=1.0; adj[d,s]=1.0
        if graph.has_edge(str(row.source_host),str(row.destination_host)):
            graph[str(row.source_host)][str(row.destination_host)]["events"] += 1
        else:
            graph.add_edge(str(row.source_host),str(row.destination_host),events=1)
    adj_norm=adj/np.maximum(adj.sum(axis=1,keepdims=True),1.0)
    return {"hosts":hosts,"host_to_idx":host_to_idx,"graph":graph,
        "node_x":torch.tensor(node_x,dtype=torch.float32),"adj_norm":torch.tensor(adj_norm,dtype=torch.float32)}

def event_indices(df: pd.DataFrame, host_to_idx: dict[str,int]):
    src=torch.tensor([host_to_idx[str(v)] for v in df.source_host],dtype=torch.long)
    dst=torch.tensor([host_to_idx[str(v)] for v in df.destination_host],dtype=torch.long)
    return src,dst


def build_time_window_graphs(df: pd.DataFrame, window_minutes: int = 60) -> dict[str, nx.DiGraph]:
    """Build one directed NetworkX graph per fixed time window for temporal inspection.

    Window graphs are intended for visualization/investigation and can also support
    future temporal-GNN extensions. The current classifier uses training-history context.
    """
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    if df.empty:
        return {}
    frame=df.copy()
    if pd.api.types.is_numeric_dtype(frame["timestamp"]):
        seconds=pd.to_numeric(frame["timestamp"],errors="coerce")
    else:
        seconds=pd.to_datetime(frame["timestamp"],errors="coerce",utc=True).astype("int64")/1e9
    frame["_window"]=(seconds // (window_minutes*60)).astype("Int64")
    result={}
    for key, part in frame.dropna(subset=["_window"]).groupby("_window",sort=True):
        graph=nx.DiGraph()
        for _,row in part.iterrows():
            s,d=str(row["source_host"]),str(row["destination_host"])
            if graph.has_edge(s,d): graph[s][d]["events"]+=1
            else: graph.add_edge(s,d,events=1)
        result[str(key)]=graph
    return result
