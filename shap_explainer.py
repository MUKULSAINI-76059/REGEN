import numpy as np
import shap
import plotly.graph_objects as go

NAMES = {"nwp_rainfall": "Model rain guess", "u850": "West wind (U850)", "v850": "South wind (V850)",
         "z500": "Upper air height (Z500)", "mslp": "Air pressure (MSLP)", "pw": "Water in air (PW)",
         "cape": "Storm energy (CAPE)", "past_rain": "Rain in last 5 days",
         "elevation": "Height of land", "lat": "North-South place", "lon": "East-West place"}


def build_explainers(bundle):
    return [shap.TreeExplainer(m) for m in bundle["experts"]]


def explain_point(explainers, x_row, proba):
    """SHAP values of the mixed answer = regime-probability-weighted SHAP of each expert."""
    vals, base = np.zeros(x_row.shape[1]), 0.0
    for p, ex in zip(proba, explainers):
        vals += p * np.ravel(ex.shap_values(x_row)[0])
        base += p * float(np.ravel(ex.expected_value)[0])
    return vals, base


def top_features(vals, x_norm, cols, k=5):
    out = []
    for i in np.argsort(-np.abs(vals))[:k]:
        lvl = "High" if x_norm[i] > 0.6 else "Low" if x_norm[i] < 0.4 else "Normal"
        out.append(f"{lvl} {NAMES[cols[i]]}: {vals[i]:+.0f} mm")
    return out


def waterfall_fig(vals, base, cols, k=5):
    order = np.argsort(-np.abs(vals))
    top, rest = order[:k], order[k:]
    labels = ["Average day"] + [NAMES[cols[i]] for i in top] + ["All other things", "Corrected rain"]
    y = [base] + [vals[i] for i in top] + [vals[rest].sum(), None]
    fig = go.Figure(go.Waterfall(x=labels, y=y, measure=["absolute"] + ["relative"] * (k + 1) + ["total"],
                                 increasing=dict(marker=dict(color="#C8E6C9")),
                                 decreasing=dict(marker=dict(color="#FFCDD2")),
                                 totals=dict(marker=dict(color="#B3E5FC")),
                                 connector=dict(line=dict(color="#B0BEC5"))))
    fig.update_layout(yaxis_title="Rain (mm)", template="plotly_white", height=420,
                      margin=dict(l=10, r=10, t=30, b=10), font=dict(size=14))
    return fig
