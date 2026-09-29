<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Poppins&weight=900&size=44&duration=2500&pause=600&color=3B82F6&center=true&vCenter=true&width=950&height=110&lines=🌧️+REGEN;🧭+Regime-Aware+Rainfall+Forecasting;🎯+Smarter+Monsoon+Predictions+with+AI;🔍+Explainable+%7C+Reliable+%7C+Uncertainty-Aware;🚀+Built+for+SIH+26080)](https://git.io/typing-svg)

---

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge" />
&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge" />
&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/github/stars/MUKULSAINI-76059/REGEN?style=for-the-badge&color=FFD700" />

<br/><br/>

> *"Don't just predict the rain — understand the weather regime behind it."*

</div>

---

## 🌦️ About the Project


**REGEN-AI** is a working prototype built for **Smart India Hackathon (SIH) — Problem Statement 26080**.

Raw NWP (Numerical Weather Prediction) rainfall forecasts often miss badly during the Indian monsoon, because one single correction cannot fit every weather situation. REGEN-AI fixes this by first identifying the **weather regime** of each grid point, then applying a **specialised expert model** for that regime.

The result is a corrected, explainable, uncertainty-aware rainfall forecast for every grid point and district of India.

> ⚠️ This prototype runs fully **offline on synthetic (mock) data** generated in `data_generator.py`. It demonstrates the pipeline end-to-end; it is not trained on real IMD/NWP data yet.

---


## ✨ Features


| Feature | Description |
| --- | --- |
| 🧭 **Regime Detection** | LightGBM classifier gives soft probabilities for 4 regimes: *Active, Break, Low Pressure, Orographic* |
| 🎯 **Regime-wise Experts** | One XGBoost regressor per regime; final forecast = probability-weighted mix of experts |
| ⛈️ **Heavy Rain Alerts** | Three binary classifiers for heavy / very heavy / extremely heavy rain (64.5, 115.6, 204.5 mm) |
| 📏 **Uncertainty Band** | 10th–90th percentile quantile models around every forecast |
| 🗺️ **Spatial Fix** | Post-processing step to correct rainfall placement (`spatial.py`) |
| 🔍 **Explainability (SHAP)** | Per-point breakdown of the top drivers behind a forecast |
| 📊 **Verification** | Raw vs corrected metrics, reliability diagram, and 10–90 coverage on held-out data |
| 🏘️ **District View** | Grid forecasts aggregated (area-weighted) into 25 districts |

---


## 🌐 Live Demo


👉 **<https://comfy-squirrel-123dca.netlify.app/>**

---

## 🧠 How It Works

```
 Raw NWP + atmospheric features
            │
            ▼
 ┌──────────────────────────┐
 │ 1. Regime Finder (LGBM)  │ → soft probabilities p0..p3
 └──────────────────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │ 2. Regime Experts (XGB)  │ → e0..e3  (one per regime)
 └──────────────────────────┘
            │   corrected = Σ pₖ · eₖ
            ▼
 ┌──────────────────────────┐
 │ 3. Heavy-Rain Checkers   │ → h0, h1, h2
 │ 4. Quantile Models       │ → q10, q90
 │ 5. Spatial Fix           │ → fixed rainfall placement
 │ 6. SHAP Explainer        │ → why this forecast?
 └──────────────────────────┘
```

**Input features:** `nwp_rainfall`, `u850`, `v850`, `z500`, `mslp`, `pw`, `cape`, `past_rain`, `elevation`, `lat`, `lon`

---

## 🛠️ Tech Stack

| Layer | Tools |
| --- | --- |
| **Backend / API** | FastAPI, Uvicorn |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Data & Maths** | pandas, NumPy, SciPy |
| **Visualisation** | Plotly (served locally, works offline) |
| **Frontend** | Web dashboard in `frontend/` |

---

## 📁 Project Structure

```
REGEN/
├── 🗂️ frontend/           # Web dashboard UI
├── data_generator.py      # Mock data, cleaning & 0–1 normalisation, district grid
├── models.py              # Regime classifier, experts, heavy-rain & quantile models
├── spatial.py             # Rainfall placement fix
├── shap_explainer.py      # SHAP explanations for a grid point
├── verification.py        # Metrics & reliability diagram
├── main.py                # FastAPI server (trains models on startup)
└── requirements.txt
```

---

## 🔌 API Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /api/forecast?day=0` | Grid points + district-level raw vs corrected rainfall, uncertainty, heavy-rain probabilities, dominant regime |
| `GET /api/explain?day=0&grid_id=0` | SHAP breakdown and regime-wise mix for one grid point |
| `GET /api/verification` | Raw vs corrected metrics, reliability curve, coverage % |

Interactive API docs are available at `/docs` once the server is running.

---

## 🚀 Getting Started

### ✅ Prerequisites

- **Python** 3.10+
- **pip**

### ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/MUKULSAINI-76059/REGEN.git
cd REGEN
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the server**

```bash
python main.py
```

Models are trained once at startup. The server picks the first free port starting from `8000`, and prints the URL in the terminal:

```
Starting REGEN-AI API on http://localhost:8000
```

**5. Open in your browser**

```
http://localhost:8000
```

---


## 🔭 Future Scope

- Replace mock data with real NWP forecasts and IMD gridded observations
- Real district boundaries (shapefiles) instead of the 2×2 grid blocks
- Extend to more lead times and ensemble members
- Deploy as a scheduled, operational forecasting service

---

## 🤝 Contributing

```bash
# Fork → Create branch → Commit → Push → Open a PR
git checkout -b feature/AmazingFeature
git commit -m "Add: AmazingFeature"
git push origin feature/AmazingFeature
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 Author
 
<div align="center">
 
**Mukul Saini**
 
[![GitHub](https://img.shields.io/badge/GitHub-MUKULSAINI--76059-181717?style=for-the-badge&logo=github)](https://github.com/MUKULSAINI-76059)

---
</div>



<div align="center">
 
### 🌾 *Better rainfall forecasts for a monsoon-dependent nation* 🌾
 
<br/>
 
⭐ **Star this repo** if you found it helpful — it means a lot!
 
<img src="https://capsule-render.vercel.app/api?type=waving&color=3B82F6&height=100&section=footer" />
 
</div>
