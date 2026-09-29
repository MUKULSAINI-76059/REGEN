import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

THRESH = [64.5, 115.6, 204.5]


def train_all(df, X):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    y, reg = df.observed_rainfall.values, df.regime_label.values
    # 1) regime finder (soft probabilities)
    clf = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, random_state=42, verbose=-1)
    clf.fit(X.iloc[tr], reg[tr])
    # 2) one expert per regime
    experts = []
    for k in range(4):
        m = tr[reg[tr] == k]
        if len(m) < 30:
            m = tr
        experts.append(xgb.XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.08,
                                        random_state=42).fit(X.iloc[m], y[m]))
    # 3) heavy rain checkers (scale_pos_weight for rare events)
    heavy = []
    for t in THRESH:
        yb = (y[tr] > t).astype(int)
        pos = int(yb.sum())
        if pos < 2:
            heavy.append(float(yb.mean()))
            continue
        spw = np.sqrt((len(yb) - pos) / pos)
        heavy.append(xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                       scale_pos_weight=spw, eval_metric="logloss",
                                       random_state=42).fit(X.iloc[tr], yb))
    # 4) how-sure: 10th and 90th percentile
    q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=100, random_state=42,
                              verbose=-1).fit(X.iloc[tr], y[tr]) for a in (0.1, 0.9)}
    return dict(clf=clf, experts=experts, heavy=heavy, q=q, train_idx=tr, test_idx=te)


def predict_all(b, X):
    P = np.zeros((len(X), 4))
    P[:, b["clf"].classes_] = b["clf"].predict_proba(X)
    E = np.column_stack([m.predict(X) for m in b["experts"]])
    corr = np.clip((P * E).sum(1), 0, None)  # MIX ANSWERS
    out = pd.DataFrame({f"p{k}": P[:, k] for k in range(4)}, index=X.index)
    for k in range(4):
        out[f"e{k}"] = E[:, k]
    out["corrected"] = corr
    for i, m in enumerate(b["heavy"]):
        out[f"h{i}"] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
    out["q10"] = np.minimum(np.clip(b["q"][0.1].predict(X), 0, None), corr)
    out["q90"] = np.maximum(b["q"][0.9].predict(X), corr)
    return out
