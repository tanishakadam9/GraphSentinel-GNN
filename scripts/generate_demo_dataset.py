"""Regenerate the synthetic GraphSentinel sample dataset (not real enterprise data)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261026

def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2025-02-01 08:00:00")
    workstations = [f"PC{i:02d}" for i in range(1, 31)]
    servers = ["FS01", "FS02", "DB01", "WEB01", "ADM01", "DC01", "APP01"]
    normal_types = ["LOGIN", "FILE_ACCESS", "SMB", "HTTPS", "DNS"]
    rows = []
    for _ in range(1600):
        ts = start + pd.Timedelta(minutes=int(rng.integers(0, 21 * 24 * 60)))
        s = str(rng.choice(workstations))
        d = str(rng.choice(servers, p=[.22, .18, .13, .18, .06, .08, .15]))
        et = str(rng.choice(normal_types, p=[.23, .22, .18, .25, .12]))
        proto = "UDP" if et == "DNS" else "TCP"
        failed = int(rng.random() < .025)
        rows.append(dict(timestamp=ts, source_host=s, destination_host=d, event_type=et, protocol=proto,
            source_port=int(rng.integers(49152, 65536)),
            destination_port={"DNS": 53, "SMB": 445, "HTTPS": 443, "LOGIN": 3389, "FILE_ACCESS": 445}[et],
            bytes_sent=int(rng.lognormal(7.2, .8)), bytes_received=int(rng.lognormal(8.0, .9)),
            failed_login=failed, session_duration=int(rng.integers(1, 900)), label=0))
    chains = [
        ["PC04", "PC17", "FS02", "ADM01", "DC01"],
        ["PC09", "PC22", "WEB01", "DB01", "ADM01"],
        ["PC12", "PC06", "FS01", "APP01", "DC01"],
        ["PC28", "PC11", "DB01", "ADM01", "DC01"],
    ]
    # Spread attack-like chains across the whole timeline so the chronological test set
    # contains labeled events. These are synthetic labels, not real observed incidents.
    attack_days = [4, 9, 15, 19]
    for chain_i, chain in enumerate(chains):
        base = start + pd.Timedelta(days=attack_days[chain_i], hours=8 + chain_i)
        for hop, (s, d) in enumerate(zip(chain[:-1], chain[1:])):
            for attempt in range(3):
                ts = base + pd.Timedelta(minutes=hop * 14 + attempt * 2)
                et = str(rng.choice(["SSH", "RDP", "REMOTE_LOGIN", "SMB"], p=[.25, .25, .35, .15]))
                rows.append(dict(timestamp=ts, source_host=s, destination_host=d, event_type=et, protocol="TCP",
                    source_port=int(rng.integers(49152, 65536)),
                    destination_port={"SSH": 22, "RDP": 3389, "REMOTE_LOGIN": 445, "SMB": 445}[et],
                    bytes_sent=int(rng.integers(8000, 42000)), bytes_received=int(rng.integers(1000, 30000)),
                    failed_login=int(attempt == 0 and rng.random() < .55),
                    session_duration=int(rng.integers(30, 1200)), label=1))
    for _ in range(30):
        idx = int(rng.integers(0, len(rows)))
        if rows[idx]["label"] == 0:
            rows[idx]["failed_login"] = 1
            rows[idx]["event_type"] = "LOGIN"
            rows[idx]["destination_port"] = 3389
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

if __name__ == "__main__":
    out = ROOT / "data" / "demo_network_events.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} synthetic events to {out}")
    print(df["label"].value_counts().rename(index={0: "normal", 1: "synthetic suspicious"}).to_string())
