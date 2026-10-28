from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
from src.train_model import run_pipeline
p=argparse.ArgumentParser(description="Train and evaluate GraphSentinel offline")
p.add_argument("--csv",default=str(ROOT/"data/demo_network_events.csv"))
p.add_argument("--epochs",type=int,default=100); p.add_argument("--threshold",type=float,default=.5)
a=p.parse_args(); result=run_pipeline(pd.read_csv(a.csv),epochs=a.epochs,threshold=a.threshold)
print("GNN metrics:",result["gnn_metrics"]); print("Baseline metrics:",result["baseline_metrics"])
print("Predictions:",ROOT/"outputs/sample_predictions.csv")
