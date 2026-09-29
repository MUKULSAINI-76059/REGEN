import numpy as np
import pandas as pd

LATS = np.linspace(8.5, 33, 10)
LONS = np.linspace(69, 95, 10)
DLAT, DLON = LATS[1] - LATS[0], LONS[1] - LONS[0]
N_DAYS = 30
REGIMES = ["Active", "Break", "Low Pressure", "Orographic"]
FEATS = ["nwp_rainfall", "u850", "v850", "z500", "mslp", "pw", "cape",
         "past_rain", "elevation", "lat", "lon"]
ATMOS = ["u850", "v850", "z500", "mslp", "pw", "cape", "past_rain"]
# very simplified outline of India (lon, lat)
INDIA = [(68.2, 23.7), (72.8, 21.1), (72.9, 19.0), (73.8, 15.5), (74.9, 12.9), (76.3, 9.5),
         (77.5, 8.1), (78.2, 8.9), (79.9, 10.3), (80.3, 13.1), (80.1, 15.9), (82.3, 16.9),
         (84.9, 19.3), (86.9, 21.3), (88.9, 21.7), (89.0, 26.4), (92.5, 21.9), (94.8, 25.0),
         (97.0, 28.0), (95.0, 29.0), (91.5, 27.8), (88.0, 27.8), (84.0, 28.5), (80.2, 30.4),
         (78.8, 32.5), (79.5, 34.5), (77.0, 35.5), (74.5, 34.7), (74.0, 32.5), (74.5, 31.0),
         (72.5, 29.5), (70.5, 28.0), (70.0, 25.5), (68.2, 23.7)]


def district_polygons():
    """25 simple rectangular 'districts', each covering 2x2 grid points."""
    polys = {}
    for i in range(5):
        for j in range(5):
            polys[i * 5 + j] = dict(lon0=LONS[2 * j] - DLON / 2, lon1=LONS[2 * j + 1] + DLON / 2,
                                    lat0=LATS[2 * i] - DLAT / 2, lat1=LATS[2 * i + 1] + DLAT / 2)
    return polys


def generate_mock_data(n_days=N_DAYS, seed=42):
    rng = np.random.default_rng(seed)
    g = pd.DataFrame([(a, b) for a in LATS for b in LONS], columns=["lat", "lon"])
    g["grid_id"] = np.arange(100)
    g["district_id"] = [((r // 10) // 2) * 5 + (r % 10) // 2 for r in range(100)]
    g["elevation"] = np.clip(100 + 1800 * np.exp(-((g.lat - 31) ** 2) / 18)
                             + 900 * np.exp(-((g.lon - 75) ** 2 + (g.lat - 14) ** 2) / 12)
                             + rng.normal(0, 60, 100), 0, 4000)
    df = pd.concat([g.assign(day=d) for d in range(n_days)], ignore_index=True)
    n = len(df)
    s = rng.normal(0, 1, n_days)[df.day.values]  # how wet each day is
    df["pw"] = np.clip(45 + 8 * s + rng.normal(0, 6, n), 15, 70)
    df["mslp"] = np.clip(1004 - 3 * s + rng.normal(0, 3, n), 990, 1016)
    df["cape"] = np.clip(1200 + 400 * s + rng.normal(0, 500, n), 0, 3000)
    df["u850"] = np.clip(8 + 3 * s + rng.normal(0, 4, n), -5, 25)
    df["v850"] = np.clip(rng.normal(0, 4, n) + s, -10, 15)
    df["z500"] = 5860 + rng.normal(0, 25, n) - 8 * s
    z = lambda x: (x - x.mean()) / x.std()
    sc = np.stack([z(z(df.pw) - z(df.mslp)),
                   z(-z(df.pw) - 0.5 * z(df.cape)),
                   z(-z(df.mslp) - z(df.z500)),
                   z(1.5 * z(df.elevation) + z(df.u850))], axis=1)
    sc += rng.normal(0, 0.4, sc.shape)
    df["regime_label"] = sc.argmax(1)  # 0=Active 1=Break 2=Low 3=Orographic
    mu = np.array([110, 8, 65, 75])[df.regime_label] + 0.8 * (df.pw - 45) + 0.02 * (df.cape - 1200)
    obs = np.clip(rng.gamma(1.6, np.maximum(mu, 3) / 1.6), 0, 300)
    df["observed_rainfall"] = obs
    df["past_rain"] = np.clip(obs * 0.5 + rng.gamma(2, 10, n), 0, 200)
    nwp = 0.6 * obs + 8 + rng.normal(0, 18, n)
    nwp = np.where(df.regime_label == 3, nwp * 0.8, nwp)
    df["nwp_rainfall"] = np.clip(nwp, 0, 300)
    cols = ["day", "grid_id", "lat", "lon", "elevation", "nwp_rainfall", "u850", "v850", "z500",
            "mslp", "pw", "cape", "past_rain", "observed_rainfall", "regime_label", "district_id"]
    return df[cols]


def clean_and_normalize(df, seed=42):
    """Add random NaNs (simulating bad data), fill with mean, scale to 0-1."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    for c in ATMOS:
        d.loc[rng.random(len(d)) < 0.03, c] = np.nan
    n_missing = int(d[ATMOS].isna().sum().sum())
    d[ATMOS] = d[ATMOS].fillna(d[ATMOS].mean())
    X = (d[FEATS] - d[FEATS].min()) / (d[FEATS].max() - d[FEATS].min())
    return d, X, n_missing
