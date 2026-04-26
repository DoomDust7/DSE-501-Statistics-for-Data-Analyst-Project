import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# All paths are relative to this file — works locally and on Streamlit Cloud
_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(_ROOT, "data", "air_quality_cleaned.csv")
XLSX_PATH = os.path.join(_ROOT, "data", "AirQuality.xlsx")

SENSOR_PAIRS = {
    "CO":   ("CO(GT)",   "PT08.S1(CO)"),
    "C6H6": ("C6H6(GT)", "PT08.S2(NMHC)"),
    "NOx":  ("NOx(GT)",  "PT08.S3(NOx)"),
    "NO2":  ("NO2(GT)",  "PT08.S4(NO2)"),
}
REF_COLS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]


def load_and_clean(save_csv=True):
    df = pd.read_excel(XLSX_PATH)

    # Drop trailing unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Parse datetime — Date is already Timestamp, Time is datetime.time
    df["Datetime"] = df["Date"].apply(lambda d: d if pd.isnull(d) else d) + \
                     df["Time"].apply(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute) if hasattr(t, 'hour') else pd.NaT)
    df = df.drop(columns=["Date", "Time"]).set_index("Datetime")

    # Drop rows where datetime could not be parsed
    df = df[df.index.notna()]
    df = df.sort_index()

    # Replace -200 sentinel with NaN
    df.replace(-200, np.nan, inplace=True)

    # Drop NMHC(GT) — only ~10% valid data
    if "NMHC(GT)" in df.columns:
        df.drop(columns=["NMHC(GT)"], inplace=True)

    # Remove consecutive all-NaN downtime block (366 rows)
    mask_all_nan = df.isnull().all(axis=1)
    df = df[~mask_all_nan]

    if save_csv:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH)

    return df


def load_cleaned():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    else:
        df = load_and_clean(save_csv=True)
    return df


def compute_h1_correlations(df):
    results = {}
    for name, (ref, sensor) in SENSOR_PAIRS.items():
        subset = df[[ref, sensor]].dropna()
        r, p = stats.pearsonr(subset[ref], subset[sensor])
        X = sm.add_constant(subset[sensor])
        ols = sm.OLS(subset[ref], X).fit()
        results[name] = {"r": r, "p": p, "ref": ref, "sensor": sensor, "ols": ols, "data": subset}
    return results


def compute_h2_stratified(df, t_thresh=25.0, rh_thresh=70.0):
    rows = []
    for name, (ref, sensor) in SENSOR_PAIRS.items():
        subset = df[[ref, sensor, "T", "RH"]].dropna()
        for t_label, t_mask in [("Low T", subset["T"] <= t_thresh), ("High T", subset["T"] > t_thresh)]:
            for rh_label, rh_mask in [("Low RH", subset["RH"] <= rh_thresh), ("High RH", subset["RH"] > rh_thresh)]:
                grp = subset[t_mask & rh_mask]
                if len(grp) < 10:
                    r = np.nan
                else:
                    r, _ = stats.pearsonr(grp[ref], grp[sensor])
                rows.append({"Pollutant": name, "T_bin": t_label, "RH_bin": rh_label, "r": r, "n": len(grp)})
    return pd.DataFrame(rows)


def compute_h2_ols(df):
    results = {}
    for name, (ref, sensor) in SENSOR_PAIRS.items():
        subset = df[[ref, sensor, "T", "RH"]].dropna()
        subset = subset.copy()
        subset["T_x_sensor"] = subset["T"] * subset[sensor]
        subset["RH_x_sensor"] = subset["RH"] * subset[sensor]
        X = sm.add_constant(subset[[sensor, "T", "RH", "T_x_sensor", "RH_x_sensor"]])
        ols = sm.OLS(subset[ref], X).fit()
        results[name] = ols
    return results


def compute_h3_temporal(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month

    hourly = df.groupby("hour")[REF_COLS].mean()
    hourly_sem = df.groupby("hour")[REF_COLS].sem()

    monthly = df.groupby("month")[REF_COLS].mean()

    # ANOVA by hour
    anova_hour = {}
    for col in ["CO(GT)", "NOx(GT)"]:
        groups = [grp[col].dropna().values for _, grp in df.groupby("hour")]
        f, p = stats.f_oneway(*groups)
        anova_hour[col] = {"F": f, "p": p}

    # ANOVA by month
    anova_month = {}
    for col in ["CO(GT)", "NOx(GT)"]:
        groups = [grp[col].dropna().values for _, grp in df.groupby("month")]
        f, p = stats.f_oneway(*groups)
        anova_month[col] = {"F": f, "p": p}

    # Hour × month pivot for CO
    df["CO(GT)"] = df["CO(GT)"]
    heatmap = df.pivot_table(values="CO(GT)", index="hour", columns="month", aggfunc="mean")

    return {
        "hourly_mean": hourly, "hourly_sem": hourly_sem,
        "monthly_mean": monthly,
        "anova_hour": anova_hour, "anova_month": anova_month,
        "heatmap": heatmap,
    }


def compute_h4_pca(df):
    subset = df[REF_COLS].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    pca = PCA(n_components=4)
    scores = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=REF_COLS,
        columns=[f"PC{i+1}" for i in range(4)]
    )
    corr_matrix = subset.corr()

    return {
        "pca": pca, "scores": scores, "loadings": loadings,
        "corr": corr_matrix, "cols": REF_COLS,
        "explained": pca.explained_variance_ratio_,
    }
