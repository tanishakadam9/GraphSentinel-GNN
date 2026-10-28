from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_COLUMNS = ["timestamp", "source_host", "destination_host"]

def load_events(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the project's normalized CSV event schema."""
    return pd.read_csv(path, nrows=nrows)

def load_lanl_auth(path: str | Path, redteam_path: str | Path | None = None,
                   max_rows: int | None = None) -> pd.DataFrame:
    """Convert LANL auth.txt (headerless, comma-separated) to project schema.

    Expected fields: time, source user, source computer, destination user,
    destination computer, auth type, logon type, auth orientation, success/failure.
    `time` is seconds from the dataset's relative origin.
    """
    names = ["time", "source_user", "source_host", "destination_user", "destination_host",
             "auth_type", "logon_type", "auth_orientation", "result"]
    raw = pd.read_csv(path, header=None, names=names, nrows=max_rows, low_memory=False)
    raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
    raw = raw.dropna(subset=["time", "source_host", "destination_host"])
    # Preserve relative timeline in an explicit numeric timestamp so it remains sortable.
    out = pd.DataFrame({
        "timestamp": raw["time"], "source_host": raw["source_host"].astype(str),
        "destination_host": raw["destination_host"].astype(str),
        "event_type": raw["logon_type"].fillna("AUTH").astype(str).str.upper(),
        "protocol": "AUTH", "bytes_sent": 0, "bytes_received": 0,
        "failed_login": raw["result"].astype(str).str.lower().str.contains("fail").astype(int),
        "label": 0, "source_user": raw["source_user"].astype(str),
        "destination_user": raw["destination_user"].astype(str), "auth_type": raw["auth_type"].astype(str),
    })
    if redteam_path is not None:
        from .labeling import label_lanl_redteam
        red = pd.read_csv(redteam_path, header=None,
            names=["time", "source_user", "source_host", "destination_user", "destination_host", "redteam"],
            low_memory=False)
        out = label_lanl_redteam(out, red)
    return out

def load_lanl_flows(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Convert LANL flows.txt (headerless) to the project schema.

    Expected fields: time, duration, source computer, source port,
    destination computer, destination port, protocol, packet count, byte count.
    """
    names = ["time", "duration", "source_host", "source_port", "destination_host",
             "destination_port", "protocol", "packet_count", "byte_count"]
    raw = pd.read_csv(path, header=None, names=names, nrows=max_rows, low_memory=False)
    for col in ["time", "duration", "source_port", "destination_port", "packet_count", "byte_count"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["time", "source_host", "destination_host"])
    return pd.DataFrame({
        "timestamp": raw["time"], "source_host": raw["source_host"].astype(str),
        "destination_host": raw["destination_host"].astype(str), "event_type": "NETWORK_FLOW",
        "protocol": raw["protocol"].astype(str), "source_port": raw["source_port"],
        "destination_port": raw["destination_port"], "bytes_sent": raw["byte_count"].fillna(0),
        "bytes_received": 0, "failed_login": 0, "session_duration": raw["duration"].fillna(0), "label": 0,
    })
