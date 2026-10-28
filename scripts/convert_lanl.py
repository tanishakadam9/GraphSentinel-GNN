from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_lanl_auth, load_lanl_flows

def main():
    p=argparse.ArgumentParser(description="Convert LANL auth.txt or flows.txt into GraphSentinel CSV format")
    p.add_argument("--kind",choices=["auth","flows"],required=True)
    p.add_argument("--input",required=True,help="Path to extracted LANL file")
    p.add_argument("--output",required=True,help="Output CSV path")
    p.add_argument("--redteam",help="Optional extracted LANL redteam.txt; supported with --kind auth")
    p.add_argument("--max-rows",type=int,default=250000,help="Limit rows for laptop-sized experiments")
    a=p.parse_args()
    if a.kind=="auth": df=load_lanl_auth(a.input,a.redteam,max_rows=a.max_rows)
    else: df=load_lanl_flows(a.input,max_rows=a.max_rows)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    print(f"Saved {len(df):,} events to {out}. Review labels and schema before training.")
if __name__=="__main__": main()
