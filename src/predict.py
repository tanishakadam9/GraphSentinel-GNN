from __future__ import annotations
import pandas as pd
from .train_model import run_pipeline

def score_events(events: pd.DataFrame, threshold: float = 0.5, epochs: int = 100) -> pd.DataFrame:
    """Run the end-to-end pipeline and return scored events for offline investigation."""
    return run_pipeline(events, epochs=epochs, threshold=threshold)["events"]
