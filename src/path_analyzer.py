from __future__ import annotations
import networkx as nx
import pandas as pd

def suspicious_subgraph(events: pd.DataFrame, threshold: float = 0.5) -> nx.DiGraph:
    suspicious=events[events["gnn_score"]>=threshold]
    graph=nx.DiGraph()
    for _,row in suspicious.iterrows():
        s,d=str(row["source_host"]),str(row["destination_host"])
        if graph.has_edge(s,d):
            graph[s][d]["events"]+=1; graph[s][d]["max_score"]=max(graph[s][d]["max_score"],float(row["gnn_score"]))
        else:
            graph.add_edge(s,d,events=1,max_score=float(row["gnn_score"]),event_type=str(row.get("event_type","CONNECTION")),
                           timestamp=str(row.get("timestamp","")))
    return graph

def find_paths(graph: nx.DiGraph, max_length: int = 4, max_paths: int = 30):
    """Return bounded directed paths with at least two edges from suspicious edges."""
    paths=[]; nodes=list(graph.nodes())
    for source in nodes:
        for target in nodes:
            if source==target: continue
            try:
                for path in nx.all_simple_paths(graph,source,target,cutoff=max_length):
                    if len(path)>=3:
                        paths.append(path)
                        if len(paths)>=max_paths: return paths
            except (nx.NetworkXNoPath,nx.NodeNotFound): pass
    return paths
