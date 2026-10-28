import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.preprocess import normalize_events, add_features
from src.graph_builder import build_graph, build_time_window_graphs
from src.path_analyzer import suspicious_subgraph, find_paths

def sample():
    return pd.DataFrame([
        {"timestamp":"2025-01-01 10:00:00","source_host":"A","destination_host":"B","label":0},
        {"timestamp":"2025-01-01 10:01:00","source_host":"B","destination_host":"C","label":1},
        {"timestamp":"2025-01-01 10:02:00","source_host":"C","destination_host":"D","label":1},
    ])

def test_normalize_and_features():
    df=normalize_events(sample()); enriched,features=add_features(df)
    assert len(enriched)==3 and len(features)>=10
    assert enriched[features].isna().sum().sum()==0

def test_graph_contains_host_nodes():
    df=normalize_events(sample()); enriched,features=add_features(df)
    g=build_graph(enriched,features)
    assert set(g["hosts"])=={"A","B","C","D"}
    assert g["graph"].has_edge("A","B")

def test_suspicious_path():
    ev=sample(); ev["gnn_score"]=[.1,.9,.95]
    graph=suspicious_subgraph(ev,.5)
    assert graph.has_edge("B","C")
    assert any(p==["B","C","D"] for p in find_paths(graph))


def test_time_window_graphs():
    df = normalize_events(sample())
    graphs = build_time_window_graphs(df, window_minutes=60)
    assert len(graphs) >= 1
    assert all(hasattr(g, "number_of_edges") for g in graphs.values())
