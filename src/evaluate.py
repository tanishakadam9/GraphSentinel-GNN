from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, roc_auc_score

def calculate_metrics(y_true, probabilities, threshold: float = 0.5):
    y=np.asarray(y_true,dtype=int); p=np.asarray(probabilities,dtype=float)
    pred=(p>=threshold).astype(int)
    result={"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    result["pr_auc"]=float(average_precision_score(y,p)) if len(np.unique(y))>1 else 0.0
    result["roc_auc"]=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else 0.0
    return result
