from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.ensemble import RandomForestClassifier
from .preprocess import normalize_events, add_features
from .graph_builder import build_graph, event_indices
from .gnn_model import GraphSAGEEdgeClassifier
from .evaluate import calculate_metrics

SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)

def _safe_split(n: int):
    """Chronological train/validation/test split, avoiding random future-to-past leakage."""
    if n < 10:
        idx=np.arange(n); return idx,idx,idx
    train_end=max(1,int(n*0.70)); val_end=max(train_end+1,int(n*0.80))
    val_end=min(val_end,n-1)
    train=np.arange(0,train_end); val=np.arange(train_end,val_end); test=np.arange(val_end,n)
    if len(test)==0: test=val.copy()
    return train,val,test

def run_pipeline(raw_df: pd.DataFrame, epochs: int = 100, threshold: float = 0.5, save_artifacts: bool = True):
    df=normalize_events(raw_df)
    enriched,event_features=add_features(df)
    n=len(enriched); train_idx,val_idx,test_idx=_safe_split(n)
    all_hosts=sorted(set(enriched.source_host.astype(str))|set(enriched.destination_host.astype(str)))
    # Build neighborhood context from training history only; test edges do not create graph edges.
    train_graph=build_graph(enriched.iloc[train_idx],event_features,all_hosts=all_hosts)
    x=train_graph["node_x"]; adj=train_graph["adj_norm"]
    src,dst=event_indices(enriched,train_graph["host_to_idx"])
    edge_x=torch.tensor(enriched[event_features].to_numpy(dtype=np.float32),dtype=torch.float32)
    y=enriched.label.to_numpy(dtype=np.int64); y_tensor=torch.tensor(y,dtype=torch.float32)
    model=GraphSAGEEdgeClassifier(node_features=x.shape[1],edge_features=edge_x.shape[1])
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4)
    train_y=y[train_idx]; pos=max(1,int((train_y==1).sum())); neg=max(1,int((train_y==0).sum()))
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(neg/pos,20.0)],dtype=torch.float32))
    history=[]
    for epoch in range(max(10,int(epochs))):
        model.train(); optimizer.zero_grad()
        logits=model(x,adj,src,dst,edge_x)
        loss=loss_fn(logits[train_idx],y_tensor[train_idx]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step()
        history.append({"epoch":epoch+1,"loss":float(loss.detach().item())})
    model.eval()
    with torch.no_grad(): probs=torch.sigmoid(model(x,adj,src,dst,edge_x)).cpu().numpy()
    rf_features=enriched[event_features].copy()
    rf=RandomForestClassifier(n_estimators=120,class_weight="balanced",random_state=SEED,min_samples_leaf=2)
    rf.fit(rf_features.iloc[train_idx],y[train_idx])
    rf_probs=rf.predict_proba(rf_features)[:,1] if len(rf.classes_)>1 else np.zeros(n)
    # If chronological test contains one class, metrics still work; disclose test counts in the UI/report.
    gnn_metrics=calculate_metrics(y[test_idx],probs[test_idx],threshold)
    rf_metrics=calculate_metrics(y[test_idx],rf_probs[test_idx],0.5)
    results=enriched.copy(); results["gnn_score"]=probs
    results["prediction"]=(probs>=threshold).astype(int)
    results["prediction_label"]=np.where(results.prediction==1,"Suspicious","Normal")
    results["baseline_score"]=rf_probs; results["is_test_row"]=False; results.loc[test_idx,"is_test_row"]=True
    # Degree features are based on the training graph only.
    out_deg=dict(train_graph["graph"].out_degree()); in_deg=dict(train_graph["graph"].in_degree())
    results["source_degree"]=results.source_host.map(out_deg).fillna(0).astype(int)
    results["destination_degree"]=results.destination_host.map(in_deg).fillna(0).astype(int)
    results["risk_level"]=pd.cut(results.gnn_score,[-0.001,0.35,0.65,1.0],labels=["Low","Medium","High"]).astype(str)
    display_graph=nx.DiGraph()
    for _, row in enriched.iterrows():
        s,d=str(row.source_host),str(row.destination_host)
        if display_graph.has_edge(s,d): display_graph[s][d]["events"] += 1
        else: display_graph.add_edge(s,d,events=1)
    result={"events":results,"graph":display_graph,"hosts":all_hosts,
        "gnn_metrics":gnn_metrics,"baseline_metrics":rf_metrics,"history":pd.DataFrame(history),
        "model":model,"threshold":threshold,"feature_cols":event_features,"node_count":len(all_hosts),
        "edge_count":len(results),"positive_count":int(y.sum()),"test_count":len(test_idx),
        "train_count":len(train_idx),"validation_count":len(val_idx),"test_indices":test_idx.tolist(),
        "label_counts":{"normal":int((y==0).sum()),"suspicious":int((y==1).sum())}}
    if save_artifacts:
        root=Path(__file__).resolve().parents[1]; models=root/"models"; outputs=root/"outputs"
        models.mkdir(exist_ok=True); outputs.mkdir(exist_ok=True)
        torch.save({"state_dict":model.state_dict(),"node_features":int(x.shape[1]),"edge_features":int(edge_x.shape[1]),
                    "feature_cols":event_features,"threshold":float(threshold),"architecture":"GraphSAGEEdgeClassifier",
                    "note":"Trained artifact for this dataset feature schema; retrain on your own data."},models/"graphsage_edge_model.pt")
        (models/"model_config.json").write_text(json.dumps({"architecture":"GraphSAGEEdgeClassifier","node_features":int(x.shape[1]),
            "edge_features":int(edge_x.shape[1]),"feature_cols":event_features,"threshold":float(threshold)},indent=2),encoding="utf-8")
        results.to_csv(outputs/"sample_predictions.csv",index=False)
        (outputs/"metrics.json").write_text(json.dumps({"gnn":gnn_metrics,"random_forest":rf_metrics,
            "split":{"train":len(train_idx),"validation":len(val_idx),"test":len(test_idx),"strategy":"chronological"},
            "label_counts":result["label_counts"]},indent=2),encoding="utf-8")
    return result
