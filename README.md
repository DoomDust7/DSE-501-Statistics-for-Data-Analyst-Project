# DSE 501 — Statistics for Data Analysts: Air Quality Project

> **Course:** DSE 501 · Statistics for Data Analysts  
> **Dataset:** [UCI Air Quality](https://archive.ics.uci.edu/ml/datasets/Air+Quality) — 9,357 hourly observations · March 2004 – April 2005 · Italian urban site  
> **Deliverables:** Jupyter Notebook · Streamlit Live Dashboard · Tableau Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doomdust7-dse-501-statistics-for-data-a-streamlit-appapp-qcrjkb.streamlit.app/)

🚀 **[Live Dashboard → doomdust7-dse-501-statistics-for-data-a-streamlit-appapp-qcrjkb.streamlit.app](https://doomdust7-dse-501-statistics-for-data-a-streamlit-appapp-qcrjkb.streamlit.app/)**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research Hypotheses](#research-hypotheses)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Project](#running-the-project)
   - [Jupyter Notebook](#1-jupyter-notebook)
   - [Streamlit Dashboard](#2-streamlit-live-dashboard)
   - [Tableau Dashboard](#3-tableau-dashboard)
6. [Technical Details](#technical-details)
   - [Data Cleaning Pipeline](#data-cleaning-pipeline)
   - [Analysis Modules](#analysis-modules)
   - [Dashboard Architecture](#dashboard-architecture)
7. [Key Findings](#key-findings)
8. [Dependencies](#dependencies)

---

## Project Overview

This project performs a comprehensive statistical analysis of co-located MOx chemical sensor arrays and certified reference instruments deployed at a road-level site in a polluted Italian city. The central questions are:

- **How accurately do low-cost MOx sensors track certified reference measurements?**
- **Do ambient temperature and humidity degrade sensor accuracy?**
- **Do pollutant concentrations follow predictable diurnal and seasonal cycles?**
- **Do traffic-source pollutants (CO, NOx, benzene) form a statistically distinct group from secondary pollutants (NO₂)?**

The analysis is structured around four formal statistical hypotheses, tested with OLS regression, one-way ANOVA, Pearson correlation, and Principal Component Analysis.

---

## Research Hypotheses

| ID | Hypothesis | Method |
|----|-----------|--------|
| **H1** | C6H6 & CO sensors: r > 0.80; NOx & NO2 sensors: r < 0.70 | Pearson r · OLS regression |
| **H2** | High T (> 25°C) and high RH (> 70%) degrade sensor accuracy | OLS with interaction terms · Stratified r |
| **H3** | Concentrations peak at rush hours (07–09h, 17–19h) and are higher in winter | One-way ANOVA by hour and month |
| **H4** | CO, NOx, C6H6 co-vary (r > 0.60); NO2 loads on a separate PCA component | Pearson correlation matrix · PCA biplot |

---

## Project Structure

```
DSE-501-Statistics-for-Data-Analyst-Project/
│
├── AirQuality_Analysis.ipynb      # Full analysis notebook (8 sections)
│
├── streamlit_app/
│   ├── app.py                     # Streamlit dashboard (5 interactive tabs)
│   ├── analysis.py                # Shared computation module
│   └── requirements.txt           # Python dependencies
│
├── tableau/
│   ├── AirQuality_Dashboard.twbx  # Packaged Tableau workbook (open in Tableau Desktop)
│   └── generate_tableau.py        # Script to regenerate .twbx from cleaned data
│
├── figures/                       # Auto-generated plots (PNG) from the notebook
│   ├── eda_missing.png
│   ├── eda_distributions.png
│   ├── eda_scatter_matrix.png
│   ├── h1_sensor_accuracy.png
│   ├── h2_confounder_heatmap.png
│   ├── h3_diurnal.png
│   ├── h3_seasonal.png
│   ├── h3_heatmap.png
│   ├── h4_correlation.png
│   └── h4_pca_biplot.png
│
├── .streamlit/
│   └── config.toml                # Dark-mode theme config
│
└── README.md
```

> **Note:** The raw dataset (`AirQuality.xlsx`) and cleaned CSV (`data/air_quality_cleaned.csv`) are not committed to this repo.  
> Download `AirQuality.xlsx` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality) and place it at:
> ```
> /Users/<your-username>/Downloads/Projects/AirQuality.xlsx
> ```
> Or update `XLSX_PATH` in `streamlit_app/analysis.py` to your local path.

---

## Setup & Installation

### Prerequisites
- Python 3.9+ (Anaconda recommended)
- Tableau Desktop (for the `.twbx` dashboard)
- `git`

### 1. Clone the repository
```bash
git clone https://github.com/DoomDust7/DSE-501-Statistics-for-Data-Analyst-Project.git
cd DSE-501-Statistics-for-Data-Analyst-Project
```

### 2. Install Python dependencies
```bash
pip install -r streamlit_app/requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn \
            streamlit plotly openpyxl tableauhyperapi nbformat
```

### 3. Place the dataset
Download `AirQuality.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality) and update the path in `streamlit_app/analysis.py`:

```python
# Line 12 in analysis.py — update this to your local path:
XLSX_PATH = "/path/to/your/AirQuality.xlsx"
```

---

## Running the Project

### 1. Jupyter Notebook

The notebook contains all 8 analysis sections with inline plots and a final results summary table.

```bash
cd DSE-501-Statistics-for-Data-Analyst-Project
jupyter notebook AirQuality_Analysis.ipynb
```

Then in the browser: **Cell → Run All**

**Sections:**
| Section | Content |
|---------|---------|
| 1 | Data Cleaning — replace -200 sentinels, drop downtime block, remove sparse NMHC(GT) |
| 2 | Exploratory Data Analysis — distributions, missing value heatmap, scatter matrix |
| 3 | Time Series Overview — 13-month dual-axis plot, monthly box plots |
| 4 | H1 Sensor Accuracy — OLS scatter plots, r-value annotation per pollutant pair |
| 5 | H2 Confounders — interaction OLS, stratified r heatmap |
| 6 | H3 Temporal Patterns — diurnal line plot, seasonal bars, hour×month heatmap, ANOVA |
| 7 | H4 PCA Structure — correlation matrix, PCA biplot, scree plot |
| 8 | Results Summary — auto-generated hypothesis outcome table |

---

### 2. Streamlit Live Dashboard

A fully interactive 5-tab dashboard with sidebar controls.

```bash
cd DSE-501-Statistics-for-Data-Analyst-Project
streamlit run streamlit_app/app.py
```

Opens at **http://localhost:8501**

**Tabs:**

| Tab | Content |
|-----|---------|
| 📈 Time Series | Dual-axis reference vs sensor plot with rolling mean, monthly box plot |
| 🎯 H1 Sensor Accuracy | Pearson r cards, regression scatter, OLS summary table |
| 🌡️ H2 Confounders | Stratified r heatmaps (react to sidebar sliders), interaction coefficient table |
| 🕐 H3 Temporal | ANOVA results table, diurnal ±95% CI, seasonal bar chart, hour×month heatmap |
| 🔬 H4 PCA | Correlation matrix, scree plot, interactive PCA biplot with loading vectors |

**Sidebar controls:**
- **Pollutant** — switches the active sensor pair across all relevant views
- **Date window** — filters all analyses to a specific time range
- **Trend smoothing** — rolling mean window in days (Time Series tab)
- **T / RH thresholds** — dynamically updates H2 stratified heatmaps

---

### 3. Tableau Dashboard

Open the packaged workbook directly in Tableau Desktop:

```
tableau/AirQuality_Dashboard.twbx
```

**Worksheets:**
| Sheet | Chart type |
|-------|-----------|
| Time Series | Dual-axis line chart — CO(GT) vs PT08.S1(CO) over time |
| Sensor Accuracy | Scatter plot by pollutant pair, color-coded by r-value tier |
| Temporal Heatmap | Hour × month heatmap for CO(GT) concentration |
| PCA Structure | PC1 vs PC2 scatter with month color encoding |

**Regenerate `.twbx` after data changes:**
```bash
python tableau/generate_tableau.py
```

---

## Technical Details

### Data Cleaning Pipeline

**Raw issues in `AirQuality.xlsx`:**

| Issue | Treatment |
|-------|-----------|
| Missing values encoded as `-200` | Replaced with `NaN` |
| 366-row consecutive device downtime block | Dropped entirely (not imputed) |
| `NMHC(GT)` — only 9.8% valid readings | Column removed from analysis |
| Date + Time in separate columns | Combined into a single `DatetimeIndex` |

**Result:** 9,357 → **9,326** usable hourly observations across 12 variables.

---

### Analysis Modules

All statistical functions live in `streamlit_app/analysis.py` and are shared between the notebook and the Streamlit app.

#### `load_and_clean(save_csv=True)`
Loads the raw Excel file, applies all cleaning steps, saves `data/air_quality_cleaned.csv`, and returns a sorted `DatetimeIndex` DataFrame.

#### `compute_h1_correlations(df)`
For each sensor pair (CO, C6H6, NOx, NO2):
- Computes Pearson r and p-value via `scipy.stats.pearsonr`
- Fits OLS model `reference ~ sensor` using `statsmodels.OLS`
- Returns dict with `{name: {r, p, ols, data, ref, sensor}}`

#### `compute_h2_stratified(df, t_thresh, rh_thresh)`
Splits observations into 4 strata: {Low T, High T} × {Low RH, High RH}.  
Computes Pearson r within each stratum for all 4 sensor pairs.

#### `compute_h2_ols(df)`
Fits interaction OLS for each pair:  
`reference ~ sensor + T + RH + T×sensor + RH×sensor`  
Returns `statsmodels` OLS results per pollutant.

#### `compute_h3_temporal(df)`
- Hourly mean ± SEM for all reference pollutants
- Monthly mean aggregations
- One-way ANOVA (`scipy.stats.f_oneway`) by hour and by month
- Hour × month pivot table for CO(GT)

#### `compute_h4_pca(df)`
- Standardizes reference pollutants with `sklearn.StandardScaler`
- Fits 4-component PCA with `sklearn.PCA`
- Returns scores, loadings DataFrame, explained variance, and Pearson correlation matrix

---

### Dashboard Architecture

```
AirQuality.xlsx
      │
      ▼
analysis.py ─── load_and_clean()
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
AirQuality_Analysis.ipynb        streamlit_app/app.py
(static, rendered notebook)      (interactive dashboard)
      │
      ▼
figures/*.png

analysis.py ──► generate_tableau.py ──► AirQuality_Dashboard.twbx
```

All three deliverables share the same `analysis.py` functions, ensuring statistical consistency across outputs.

---

## Key Findings

| Hypothesis | Result |
|-----------|--------|
| H1 — C6H6 sensor accuracy | ✅ Supported — r = +0.982 (well above 0.80 threshold) |
| H1 — CO sensor accuracy | ✅ Supported — r = +0.879 |
| H1 — NOx sensor accuracy | ✅ Supported — r = −0.656 (below 0.70, cross-sensitivity confirmed) |
| H1 — NO2 sensor accuracy | ✅ Supported — r = +0.158 (extremely low, secondary pollutant effect) |
| H2 — T/RH confounders | ✅ Supported — significant interaction terms in OLS (p < 0.05 for most pairs) |
| H3 — Diurnal ANOVA | ✅ Supported — F >> 1, p ≈ 0 across all pollutants |
| H3 — Seasonal ANOVA | ✅ Supported — significantly higher concentrations in winter months |
| H4 — NO2 diverges in PCA | ✅ Supported — NO2 loads strongly on PC2 while CO/NOx/C6H6 cluster on PC1 |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling |
| `matplotlib`, `seaborn` | Static plots (notebook/figures) |
| `plotly` | Interactive charts (Streamlit) |
| `statsmodels` | OLS regression |
| `scipy` | Pearson r, one-way ANOVA |
| `scikit-learn` | PCA, StandardScaler |
| `streamlit` | Live dashboard framework |
| `tableauhyperapi` | Tableau Hyper extract generation |
| `openpyxl` | Excel file reading |

---

*DSE 501 — Statistics for Data Analysts · Arizona State University*
