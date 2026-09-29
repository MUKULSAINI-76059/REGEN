import numpy as np
import pandas as pd


def metrics(obs, pred):
    d = np.asarray(pred) - np.asarray(obs)
    return dict(RMSE=float(np.sqrt(np.mean(d ** 2))), MAE=float(np.mean(np.abs(d))), Bias=float(d.mean()))


def reliability(prob, event, bins=5):
    prob, event = np.asarray(prob), np.asarray(event).astype(float)
    b = np.clip((prob * bins).astype(int), 0, bins - 1)
    rows = [(prob[b == i].mean(), event[b == i].mean(), int((b == i).sum())) for i in range(bins) if (b == i).any()]
    return pd.DataFrame(rows, columns=["Chance we said", "How often it happened", "Cases"])
