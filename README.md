<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Poppins&weight=900&size=44&duration=2500&pause=600&color=3B82F6&center=true&vCenter=true&width=950&height=110&lines=🌧️+REGEN;🧭+Regime-Aware+Rainfall+Forecasting;🎯+Smarter+Monsoon+Predictions+with+AI;🔍+Explainable+%7C+Reliable+%7C+Uncertainty;🚀+Built+for+SIH+26080)](https://git.io/typing-svg)

---

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge" />
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

👉 **[REGEN-AI Live Demo](https://regen-w.netlify.app/)**

---

## 🧠 How It Works

```text
Raw NWP + atmospheric features
           │
           ▼
┌──────────────────────────┐
│ 1. Regime Finder (LGBM)  │ → soft probabilities p0..p3
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 2. Regime Experts (XGB)  │ → e0..e3
│    One expert per regime │
└──────────────────────────┘
           │
           │ corrected = Σ pₖ · eₖ
           ▼
┌──────────────────────────┐
│ 3. Heavy-Rain Checkers   │ → h0, h1, h2
│ 4. Quantile Models       │ → q10, q90
│ 5. Spatial Fix           │ → fixed rainfall placement
│ 6. SHAP Explainer        │ → why this forecast?
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│     Final Forecast       │
│ Grid + District Level    │
└──────────────────────────┘
```

### Input Features

```text
nwp_rainfall
u850
v850
z500
mslp
pw
cape
past_rain
elevation
lat
lon
```

---

## 🛠️ Tech Stack

| Layer | Tools |
| --- | --- |
| **Backend / API** | FastAPI, Uvicorn |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Data & Maths** | pandas, NumPy, SciPy |
| **Visualisation** | Plotly |
| **Frontend** | HTML, CSS, JavaScript |
| **Data** | Synthetic / Mock NWP & rainfall data |

---

## 📁 Project Structure

```text
REGEN/
│
├── 🗂️ frontend/
│   ├── Page img/
│   │   ├── Dashboard Page (2).png
│   │   ├── Dashboard Page.png
│   │   ├── Verification Page.png
│   │   └── home page.png
│   │
│   ├── logo/
│   │   ├── logo.webp
│   │   └── map.webp
│   │
│   └── index.html
│
├── data_generator.py
│   # Generates synthetic weather/rainfall data
│
├── models.py
│   # Regime classifier
│   # Regime-wise expert models
│   # Heavy-rain classifiers
│   # Quantile models
│
├── spatial.py
│   # Spatial rainfall placement correction
│
├── shap_explainer.py
│   # SHAP-based forecast explanations
│
├── verification.py
│   # Model verification
│   # RMSE, MAE, Bias
│   # Reliability curve
│   # Coverage calculation
│
├── main.py
│   # FastAPI application
│   # API endpoints
│   # Model initialization
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/forecast?day=0` | Returns grid-level and district-level rainfall forecasts |
| `GET` | `/api/explain?day=0&grid_id=0` | Returns SHAP explanation and regime-wise forecast contribution |
| `GET` | `/api/verification` | Returns verification metrics, reliability curve and coverage |

### Forecast API

```text
GET /api/forecast?day=0
```

Provides:

- Raw rainfall
- AI-corrected rainfall
- Lower uncertainty bound
- Upper uncertainty bound
- Heavy rainfall probabilities
- Dominant weather regime
- Grid-level forecast
- District-level forecast

### Explainability API

```text
GET /api/explain?day=0&grid_id=0
```

Provides:

- SHAP feature contribution
- Important weather features
- Regime probabilities
- Regime-wise expert contribution

### Verification API

```text
GET /api/verification
```

Provides:

- RMSE
- MAE
- Bias
- Raw forecast metrics
- Corrected forecast metrics
- Reliability curve
- Coverage percentage
- Test case count

Interactive API documentation is available at:

```text
/docs
```

when the FastAPI server is running.

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have:

- **Python 3.10+**
- **pip**
- **Git**

---

### ⚙️ Installation

#### 1. Clone the repository

```bash
git clone https://github.com/MUKULSAINI-76059/REGEN.git
cd REGEN
```

---

#### 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

#### 4. Run the FastAPI server

```bash
python main.py
```

The application starts the model pipeline and launches the FastAPI server.

You should see something similar to:

```text
Starting REGEN-AI API on http://localhost:8000
```

---

#### 5. Open the application

Open:

```text
http://localhost:8000
```

---

### 📚 API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://localhost:8000/docs
```

---

## 📸 Screenshots

### 🏠 Home Page

<img width="1852" height="889" alt="Home Page" src="./frontend/Page%20img/home%20page.png" />

---

### 📊 Dashboard

<img width="1894" height="876" alt="Dashboard" src="./frontend/Page%20img/Dashboard%20Page.png" />

---

### ✅ Verification

<img width="1898" height="886" alt="Verification" src="./frontend/Page%20img/Verification%20Page.png" />

---

## 📈 Verification

REGEN-AI provides a dedicated verification dashboard to compare:

- Raw NWP forecast
- AI-corrected forecast
- RMSE
- MAE
- Bias
- Reliability
- Prediction interval coverage

The verification module helps analyse how the corrected rainfall forecast behaves compared with the original forecast.

---

## 🧭 Weather Regimes

REGEN-AI classifies each grid point into one of four weather regimes:

| Regime | Description |
| --- | --- |
| ☀️ **Active** | Active monsoon rainfall conditions |
| 🌤️ **Break** | Relatively suppressed rainfall conditions |
| 🌀 **Low Pressure** | Rainfall influenced by low-pressure systems |
| ⛰️ **Orographic** | Rainfall influenced by terrain/elevation |

The model does not use one single correction model for every weather situation. Instead, the regime probabilities are used to combine specialised regime-wise experts.

---

## 🎯 AI Correction Pipeline

The core correction follows:

```text
NWP Rainfall
     │
     ▼
Weather Regime Detection
     │
     ├──────────────┐
     │              │
     ▼              ▼
Active Expert   Break Expert
     │              │
     ├──────────────┤
     │              │
     ▼              ▼
Low Pressure    Orographic
   Expert          Expert
     │              │
     └───────┬──────┘
             ▼
   Probability Weighted
        Combination
             │
             ▼
      Corrected Rainfall
```

---

## ⛈️ Heavy Rain Detection

The system also estimates probabilities for different rainfall severity levels.

| Category | Threshold |
| --- | ---: |
| **Heavy Rain** | 64.5 mm |
| **Very Heavy Rain** | 115.6 mm |
| **Extremely Heavy Rain** | 204.5 mm |

These probabilities can be used to identify areas that may experience significant rainfall.

---

## 📏 Uncertainty Estimation

REGEN-AI does not provide only a single rainfall value.

It also generates an uncertainty interval using:

```text
q10 → Lower rainfall estimate
q50 → Main forecast
q90 → Upper rainfall estimate
```

This gives the user an indication of the range around the predicted rainfall.

---

## 🔍 Explainability

The system uses **SHAP** to explain individual predictions.

For a selected grid point, the system can show which input features contributed to the rainfall forecast.

Example features:

```text
NWP Rainfall
Past Rainfall
CAPE
Precipitable Water
MSLP
850 hPa Wind
Elevation
Latitude
Longitude
```

---

## 🗺️ Spatial Correction

Rainfall prediction is not only about the amount of rainfall.

It is also important to estimate **where rainfall occurs**.

The `spatial.py` module performs post-processing to improve the spatial placement of predicted rainfall across the grid.

---

## 🏘️ District-Level Forecast

Grid-level predictions are aggregated to provide district-level rainfall information.

The system uses an area-weighted aggregation approach to calculate rainfall for supported districts.

---

## 🔭 Future Scope

- Replace synthetic data with real NWP forecasts
- Integrate real IMD gridded rainfall observations
- Use real district boundaries and shapefiles
- Increase the number of forecast lead times
- Support ensemble forecasting
- Improve uncertainty calibration
- Add more weather regimes
- Deploy as an operational forecasting service
- Add automated scheduled forecasts
- Improve real-time monitoring and alerts

---

## 🤝 Contributing

Contributions are welcome! 🙌

### Fork the repository

```bash
git fork
```

### Create a new branch

```bash
git checkout -b feature/AmazingFeature
```

### Commit your changes

```bash
git add .
git commit -m "Add: AmazingFeature"
```

### Push your branch

```bash
git push origin feature/AmazingFeature
```

Then open a Pull Request.

---

## 📜 License

Distributed under the **MIT License**.

See the `LICENSE` file for more information.

---

## 👨‍💻 Authors

<div align="center">

### Mukul Saini

[![GitHub](https://img.shields.io/badge/GitHub-MUKULSAINI--76059-181717?style=for-the-badge&logo=github)](https://github.com/MUKULSAINI-76059)

<br/>

### Omkar Pandey

[![GitHub](https://img.shields.io/badge/GitHub-panditomkarpandey-181717?style=for-the-badge&logo=github)](https://github.com/panditomkarpandey)

</div>

---

<div align="center">

### 🌾 *Better rainfall forecasts for a monsoon-dependent nation* 🌾

<br/>

⭐ **Star this repository if you found it helpful!**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=3B82F6&height=100&section=footer" />

</div>
