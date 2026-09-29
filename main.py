import os
import socket
import numpy as np, pandas as pd, plotly
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from data_generator import *
from models import train_all, predict_all
from spatial import fix_rain_place
from shap_explainer import build_explainers, explain_point, NAMES
from verification import metrics, reliability

# ---- train everything once, when the server starts ----
print("Initializing REGEN-AI models...")
df, X, N_MISSING = clean_and_normalize(generate_mock_data())
B = train_all(df, X)
df = pd.concat([df, predict_all(B, X)], axis=1)
EX = build_explainers(B)
POLYS = district_polygons()
R = lambda x: round(float(x), 2)
app = FastAPI(title="REGEN-AI API")


def day_frame(day):
    if not 0 <= day < N_DAYS:
        raise HTTPException(404, "day out of range")
    d = df[df.day == day].sort_values("grid_id").copy()
    d["fixed"] = fix_rain_place(d.corrected.values)
    return d


@app.get("/api/forecast")
def forecast(day: int = 0):
    d = day_frame(day)
    w = np.cos(np.radians(d.lat))  # area weight
    cols = ["nwp_rainfall", "corrected", "q10", "q90", "h0", "h1", "h2", "p0", "p1", "p2", "p3"]
    agg = d[cols].mul(w, axis=0).groupby(d.district_id).sum().div(w.groupby(d.district_id).sum(), axis=0)
    dist = [dict(id=int(k), name=f"District {k + 1}", raw=R(a.nwp_rainfall), corrected=R(a.corrected), q10=R(a.q10),
                 q90=R(a.q90), heavy=[R(a.h0), R(a.h1), R(a.h2)], probs=[R(a[f"p{j}"]) for j in range(4)],
                 regime=REGIMES[int(np.argmax([a[f"p{j}"] for j in range(4)]))], poly={n: R(v) for n, v in POLYS[k].items()})
            for k, a in agg.iterrows()]
    pts = [dict(id=int(r.grid_id), lat=R(r.lat), lon=R(r.lon), raw=R(r.nwp_rainfall), cor=R(r.corrected), fix=R(r.fixed))
           for r in d.itertuples()]
    return dict(day=day, points=pts, districts=dist, grid=dict(lats=LATS.tolist(), lons=LONS.tolist()), india=INDIA, regimes=REGIMES)


@app.get("/api/explain")
def explain(day: int = 0, grid_id: int = 0):
    d = day_frame(day)
    m = d[d.grid_id == grid_id]
    if m.empty:
        raise HTTPException(404, "grid point not found")
    i = m.index[0]; row = df.loc[i]
    p = row[[f"p{k}" for k in range(4)]].values.astype(float)
    v, base = explain_point(EX, X.loc[[i]], p)
    o = np.argsort(-np.abs(v))
    feats = [dict(name=NAMES[FEATS[j]], value=R(v[j]), level="High" if X.loc[i, FEATS[j]] > .6 else "Low" if X.loc[i, FEATS[j]] < .4 else "Normal") for j in o[:5]]
    return dict(grid_id=grid_id, base=R(base), feats=feats, other=R(v[o[5:]].sum()), final=R(row.corrected), raw=R(row.nwp_rainfall),
                mix=[dict(regime=REGIMES[k], prob=R(p[k]), expert=R(row[f"e{k}"]), part=R(p[k] * row[f"e{k}"])) for k in range(4)])


@app.get("/api/verification")
def verification():
    t = df.loc[B["test_idx"]]
    rl = reliability(t.h0, t.observed_rainfall > 64.5)
    return dict(raw=metrics(t.observed_rainfall, t.nwp_rainfall), cor=metrics(t.observed_rainfall, t.corrected),
                rel=dict(x=rl.iloc[:, 0].round(3).tolist(), y=rl.iloc[:, 1].round(3).tolist()),
                cov=R(((t.observed_rainfall >= t.q10) & (t.observed_rainfall <= t.q90)).mean() * 100),
                n=len(t), missing=N_MISSING)


HERE = os.path.dirname(__file__)


@app.get("/plotly.min.js")  # served locally so the site works offline
def plotly_js():
    return FileResponse(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"))


app.mount("/", StaticFiles(directory=HERE, html=True), name="site")


def find_free_port(start_port=8000, max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts - 1}")


if __name__ == "__main__":
    port = find_free_port()
    print(f"Starting REGEN-AI API on http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
