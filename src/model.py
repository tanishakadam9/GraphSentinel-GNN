"""Public model module matching the project structure in the report.

The implementation lives in gnn_model.py to preserve the original prototype's
imports; this module provides the stable, documented import path.
"""
from .gnn_model import GraphSAGELayer, GraphSAGEEdgeClassifier

__all__ = ["GraphSAGELayer", "GraphSAGEEdgeClassifier"]
