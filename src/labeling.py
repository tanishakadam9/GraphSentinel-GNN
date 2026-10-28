from __future__ import annotations
import pandas as pd

def label_lanl_redteam(events: pd.DataFrame, redteam: pd.DataFrame, tolerance_seconds: int = 1800) -> pd.DataFrame:
    """Label events that match a LANL red-team source/destination pair near its event time.

    This is a practical weak-label join, not a claim that every unlabeled event is benign.
    """
    out = events.copy()
    out["label"] = 0
    if redteam.empty or out.empty:
        return out
    red = redteam.copy()
    for col in ["time", "source_host", "destination_host"]:
        if col not in red.columns:
            raise ValueError(f"Red-team file is missing expected column: {col}")
    red["time"] = pd.to_numeric(red["time"], errors="coerce")
    out["_time_num"] = pd.to_numeric(out["timestamp"], errors="coerce")
    red = red.dropna(subset=["time", "source_host", "destination_host"])
    # Exact host pair and a bounded time tolerance; process in groups to avoid a full cross join.
    pairs = {}
    for row in red.itertuples(index=False):
        key = (str(row.source_host), str(row.destination_host))
        pairs.setdefault(key, []).append(float(row.time))
    import numpy as np
    labels = []
    for source, destination, t in zip(out["source_host"], out["destination_host"], out["_time_num"]):
        times = pairs.get((str(source), str(destination)), [])
        labels.append(int(pd.notna(t) and bool(times) and np.min(np.abs(np.asarray(times) - float(t))) <= tolerance_seconds))
    out["label"] = labels
    return out.drop(columns=["_time_num"])

def validate_binary_labels(df: pd.DataFrame, column: str = "label") -> None:
    if column not in df:
        raise ValueError(f"Missing label column: {column}")
    vals = set(pd.to_numeric(df[column], errors="coerce").dropna().astype(int).unique())
    if not vals.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(vals)}")
