from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["timestamp", "source_host", "destination_host"]

def normalize_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize host communication events from CSV or LANL converters."""
    if raw is None or raw.empty:
        raise ValueError("The input dataset is empty.")
    df = raw.copy()
    aliases = {
        "src": "source_host", "source": "source_host", "src_host": "source_host",
        "dst": "destination_host", "destination": "destination_host", "dst_host": "destination_host",
        "time": "timestamp", "date": "timestamp", "protocol_name": "protocol", "bytes": "bytes_sent",
    }
    lowmap = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for low, original in lowmap.items():
        if low in aliases and aliases[low] not in df.columns:
            rename[original] = aliases[low]
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    # LANL converters use numeric relative seconds; regular CSVs use wall-clock timestamps.
    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.notna().mean() > 0.95:
        df["timestamp"] = ts_num
        df["_sort_time"] = ts_num
    else:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["_sort_time"] = parsed.astype("int64", errors="ignore")
        df.loc[parsed.isna(), "_sort_time"] = np.nan
    df["source_host"] = df["source_host"].astype("string").str.strip()
    df["destination_host"] = df["destination_host"].astype("string").str.strip()
    df = df.dropna(subset=REQUIRED + ["_sort_time"])
    df = df[(df.source_host != "") & (df.destination_host != "") & (df.source_host != df.destination_host)].copy()
    if df.empty:
        raise ValueError("No valid events remain after cleaning. Check timestamp and host columns.")
    defaults = {"event_type":"CONNECTION", "protocol":"UNKNOWN", "bytes_sent":0, "bytes_received":0,
                "failed_login":0, "session_duration":0, "source_port":0, "destination_port":0, "label":0}
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    for col in ["bytes_sent", "bytes_received", "failed_login", "session_duration", "source_port", "destination_port", "label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["bytes_sent", "bytes_received", "session_duration", "source_port", "destination_port"]:
        df[col] = df[col].clip(lower=0)
    df["failed_login"] = df["failed_login"].clip(0, 1).astype(int)
    df["label"] = df["label"].clip(0, 1).astype(int)
    df["event_type"] = df["event_type"].astype("string").fillna("CONNECTION").str.upper()
    df["protocol"] = df["protocol"].astype("string").fillna("UNKNOWN").str.upper()
    df = df.sort_values("_sort_time", kind="stable").drop(columns=["_sort_time"]).reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build event features without using labels or future events."""
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out["timestamp"]):
        rel = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0)
        out["hour"] = ((rel % 86400) // 3600).astype(float)
        out["day_of_week"] = ((rel // 86400) % 7).astype(float)
    else:
        ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out["hour"] = ts.dt.hour.fillna(0).astype(float)
        out["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(float)
    out["log_bytes_sent"] = np.log1p(out["bytes_sent"].astype(float))
    out["log_bytes_received"] = np.log1p(out["bytes_received"].astype(float))
    out["log_session_duration"] = np.log1p(out["session_duration"].astype(float))
    out["log_source_port"] = np.log1p(out["source_port"].astype(float)) / 12.0
    out["log_destination_port"] = np.log1p(out["destination_port"].astype(float)) / 12.0
    out["failed_login_feature"] = out["failed_login"].astype(float)
    out["event_is_remote"] = out["event_type"].str.contains("RDP|SSH|SMB|REMOTE|WINRM|LOGIN|AUTH", regex=True).astype(float)
    out["protocol_tcp"] = (out["protocol"] == "TCP").astype(float)
    out["protocol_udp"] = (out["protocol"] == "UDP").astype(float)
    # Historical novelty and cumulative activity: only previous rows influence each event.
    seen_pairs, source_counts, dest_counts = set(), {}, {}
    new_pair, prior_src, prior_dst = [], [], []
    # Causal counts within the current one-hour window; counters reset when time advances.
    if pd.api.types.is_numeric_dtype(out["timestamp"]):
        time_seconds = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        time_seconds = (pd.to_datetime(out["timestamp"], errors="coerce", utc=True).astype("int64") / 1e9).to_numpy(dtype=float)
    current_bucket = None
    pair_counts_hour, source_counts_hour, destinations_hour = {}, {}, {}
    prior_pair_hour, prior_source_hour, unique_dest_hour = [], [], []
    for s, d, sec in zip(out["source_host"].astype(str), out["destination_host"].astype(str), time_seconds):
        key = (s, d)
        new_pair.append(0.0 if key in seen_pairs else 1.0)
        prior_src.append(float(source_counts.get(s, 0)))
        prior_dst.append(float(dest_counts.get(d, 0)))
        bucket = int(sec // 3600) if np.isfinite(sec) else 0
        if bucket != current_bucket:
            current_bucket = bucket
            pair_counts_hour, source_counts_hour, destinations_hour = {}, {}, {}
        prior_pair_hour.append(float(pair_counts_hour.get(key, 0)))
        prior_source_hour.append(float(source_counts_hour.get(s, 0)))
        unique_dest_hour.append(float(len(destinations_hour.get(s, set()))))
        seen_pairs.add(key); source_counts[s] = source_counts.get(s, 0) + 1; dest_counts[d] = dest_counts.get(d, 0) + 1
        pair_counts_hour[key] = pair_counts_hour.get(key, 0) + 1
        source_counts_hour[s] = source_counts_hour.get(s, 0) + 1
        destinations_hour.setdefault(s, set()).add(d)
    out["new_destination_pair"] = new_pair
    out["log_prior_source_events"] = np.log1p(prior_src) / 10.0
    out["log_prior_destination_events"] = np.log1p(prior_dst) / 10.0
    out["log_prior_pair_events_1h"] = np.log1p(prior_pair_hour) / 10.0
    out["log_prior_source_events_1h"] = np.log1p(prior_source_hour) / 10.0
    out["unique_destinations_seen_1h"] = np.asarray(unique_dest_hour, dtype=float) / 20.0
    # Fixed scaling makes inference deterministic and avoids fitting scalers on test data.
    out["hour"] /= 23.0
    out["day_of_week"] /= 6.0
    out["log_bytes_sent"] /= 20.0
    out["log_bytes_received"] /= 20.0
    out["log_session_duration"] /= 12.0
    features = ["hour", "day_of_week", "log_bytes_sent", "log_bytes_received", "log_session_duration",
                "log_source_port", "log_destination_port", "failed_login_feature", "event_is_remote",
                "protocol_tcp", "protocol_udp", "new_destination_pair", "log_prior_source_events",
                "log_prior_destination_events", "log_prior_pair_events_1h", "log_prior_source_events_1h",
                "unique_destinations_seen_1h"]
    out[features] = out[features].replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    return out, features
