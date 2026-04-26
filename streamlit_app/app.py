import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from analysis import (
    load_cleaned, compute_h1_correlations,
    compute_h2_stratified, compute_h2_ols,
    compute_h3_temporal, compute_h4_pca,
    SENSOR_PAIRS, REF_COLS,
)

st.set_page_config(
    page_title="Air Quality Analysis",
    layout="wide",
    page_icon="🌫️",
    initial_sidebar_state="expanded",
)

# ── Dark-mode design tokens ────────────────────────────────────────────────────
BG          = "#0E1117"
CARD        = "#1A1F2E"
CARD2       = "#212638"
BORDER      = "#2D3748"
TEXT        = "#E2E8F0"
MUTED       = "#94A3B8"
PRIMARY     = "#4F8EF7"
SUCCESS     = "#34D399"
WARNING     = "#FBBF24"
DANGER      = "#F87171"

CHART_COLORS = {
    "ref":    "#4F8EF7",
    "sensor": "#FB923C",
    "co":     "#4F8EF7",
    "nox":    "#34D399",
    "no2":    "#C084FC",
    "c6h6":   "#FB923C",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1A1F2E",
    plot_bgcolor="#1A1F2E",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT),
    margin=dict(l=52, r=24, t=52, b=52),
    legend=dict(
        bgcolor="rgba(26,31,46,0.95)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=11, color=TEXT),
    ),
    hoverlabel=dict(
        bgcolor="#212638",
        bordercolor=BORDER,
        font=dict(size=12, color=TEXT),
    ),
    xaxis=dict(gridcolor="#2D3748", linecolor="#2D3748", zerolinecolor="#2D3748"),
    yaxis=dict(gridcolor="#2D3748", linecolor="#2D3748", zerolinecolor="#2D3748"),
)

def apply_layout(fig, height=420, **kwargs):
    merged = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(height=height, **merged)
    return fig

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}

/* ─ Header ─ */
.dash-header {{
    background: linear-gradient(135deg, #0F2657 0%, #1E3A8A 60%, #1D4ED8 100%);
    border: 1px solid #1E40AF;
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 20px;
}}
.dash-header h1 {{ margin:0; font-size:1.85rem; font-weight:700; color:#F0F7FF; letter-spacing:-0.5px; }}
.dash-header p  {{ margin:8px 0 0; font-size:0.88rem; color:#93C5FD; }}

/* ─ KPI grid ─ */
.kpi-grid {{ display:flex; gap:14px; margin-bottom:22px; }}
.kpi-card {{
    flex:1; background:{CARD}; border:1px solid {BORDER};
    border-radius:12px; padding:18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
}}
.kpi-label {{ font-size:0.7rem; font-weight:700; color:{MUTED}; text-transform:uppercase; letter-spacing:0.07em; }}
.kpi-value {{ font-size:1.75rem; font-weight:700; color:{TEXT}; margin:6px 0 3px; line-height:1; }}
.kpi-sub   {{ font-size:0.75rem; color:{MUTED}; }}

/* ─ Section header ─ */
.section-header {{
    font-size:0.95rem; font-weight:600; color:{TEXT};
    border-left:3px solid {PRIMARY}; padding-left:10px;
    margin:22px 0 12px; letter-spacing:0.01em;
}}

/* ─ Hypothesis card ─ */
.hypo-card {{
    background: #162044;
    border:1px solid #1E40AF;
    border-left:4px solid {PRIMARY};
    border-radius:10px;
    padding:14px 20px;
    margin-bottom:20px;
}}
.hypo-card .h-label {{ font-size:0.65rem; font-weight:700; color:#60A5FA; text-transform:uppercase; letter-spacing:0.08em; }}
.hypo-card .h-text  {{ font-size:0.88rem; color:#CBD5E1; margin-top:5px; line-height:1.55; }}

/* ─ r-value badge ─ */
.badge {{ display:inline-block; border-radius:6px; padding:3px 10px; font-size:0.75rem; font-weight:600; }}
.badge-high   {{ background:rgba(52,211,153,0.15); color:{SUCCESS}; border:1px solid rgba(52,211,153,0.3); }}
.badge-medium {{ background:rgba(251,191,36,0.15);  color:{WARNING}; border:1px solid rgba(251,191,36,0.3); }}
.badge-low    {{ background:rgba(248,113,113,0.15); color:{DANGER};  border:1px solid rgba(248,113,113,0.3); }}

/* ─ r-value card ─ */
.r-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
    padding:18px 14px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);
}}
.r-card .r-name {{ font-size:0.72rem; font-weight:700; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px; }}
.r-card .r-val  {{ font-size:1.5rem; font-weight:700; color:{TEXT}; margin-bottom:8px; }}

/* ─ ANOVA table ─ */
.anova-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
.anova-table th {{ background:{CARD2}; color:{MUTED}; font-weight:600; padding:10px 14px; border-bottom:1px solid {BORDER}; text-align:left; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; }}
.anova-table td {{ padding:10px 14px; border-bottom:1px solid {BORDER}; color:{TEXT}; }}
.anova-table tr:last-child td {{ border-bottom:none; }}
.anova-table tr:hover td {{ background:rgba(79,142,247,0.05); }}
.sig   {{ color:{SUCCESS}; font-weight:600; }}
.insig {{ color:{DANGER};  font-weight:600; }}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {{ background:#111827 !important; border-right:1px solid {BORDER}; }}
.sidebar-label {{ font-size:0.68rem; font-weight:700; color:{MUTED}; text-transform:uppercase; letter-spacing:0.08em; margin:18px 0 6px; }}
.sidebar-info {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:12px 14px; font-size:0.8rem; color:{MUTED}; line-height:1.7; }}

/* ─ Tabs ─ */
button[data-baseweb="tab"] {{ font-size:0.84rem !important; font-weight:500 !important; color:{MUTED} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{PRIMARY} !important; font-weight:600 !important; }}

/* ─ Hide branding ─ */
#MainMenu, footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_cleaned()

def hypo_card(text, label="Hypothesis"):
    st.markdown(f"""
    <div class="hypo-card">
        <div class="h-label">{label}</div>
        <div class="h-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── Data & filters ─────────────────────────────────────────────────────────────
df_full = get_data()

with st.sidebar:
    st.markdown("### ⚙️ Controls")

    st.markdown('<div class="sidebar-label">Pollutant</div>', unsafe_allow_html=True)
    pollutant = st.selectbox(
        "Active pollutant", list(SENSOR_PAIRS.keys()), label_visibility="collapsed",
        help="Drives the Time Series tab and H1 scatter default",
    )

    st.markdown('<div class="sidebar-label">Date window</div>', unsafe_allow_html=True)
    date_min = df_full.index.min().date()
    date_max = df_full.index.max().date()
    date_range = st.date_input(
        "Date window", value=(date_min, date_max),
        min_value=date_min, max_value=date_max,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Trend smoothing</div>', unsafe_allow_html=True)
    rolling_days = st.slider("Rolling mean (days)", 1, 30, 7, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">H2 — Environmental thresholds</div>', unsafe_allow_html=True)
    t_thresh  = st.slider("Temperature (°C)", 10.0, 40.0, 25.0, 0.5,
                          help="Splits obs into Low T / High T strata")
    rh_thresh = st.slider("Humidity (%)", 30.0, 100.0, 70.0, 1.0,
                          help="Splits obs into Low RH / High RH strata")

    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-info">
        <b>UCI Air Quality Dataset</b><br>
        9,357 hourly observations<br>
        March 2004 – April 2005<br>
        Road-level · Italian urban site
    </div>""", unsafe_allow_html=True)

df = df_full.loc[str(date_range[0]):str(date_range[1])] if len(date_range) == 2 else df_full


# ── Header ─────────────────────────────────────────────────────────────────────
d0 = date_range[0] if len(date_range) == 2 else date_min
d1 = date_range[1] if len(date_range) == 2 else date_max
st.markdown(f"""
<div class="dash-header">
    <h1>🌫️ Air Quality Statistical Dashboard</h1>
    <p>Sensor calibration &nbsp;·&nbsp; Temporal patterns &nbsp;·&nbsp; Inter-pollutant structure
    &nbsp;&nbsp;|&nbsp;&nbsp; {len(df):,} observations &nbsp;·&nbsp; {d0} → {d1}</p>
</div>""", unsafe_allow_html=True)

# ── KPI strip ──────────────────────────────────────────────────────────────────
missing_pct = df.isnull().mean().mean() * 100
date_span   = (df.index.max() - df.index.min()).days
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Observations</div>
        <div class="kpi-value">{len(df):,}</div>
        <div class="kpi-sub">hourly readings</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Date Span</div>
        <div class="kpi-value">{date_span}</div>
        <div class="kpi-sub">days in window</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Sensor Pairs</div>
        <div class="kpi-value">{len(SENSOR_PAIRS)}</div>
        <div class="kpi-sub">reference + MOx</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Missing</div>
        <div class="kpi-value">{missing_pct:.1f}%</div>
        <div class="kpi-sub">across channels</div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Time Series",
    "🎯  H1 · Sensor Accuracy",
    "🌡️  H2 · Confounders",
    "🕐  H3 · Temporal",
    "🔬  H4 · PCA",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    ref_col, sensor_col = SENSOR_PAIRS[pollutant]
    c_l, c_r = st.columns([4, 1])
    with c_l:
        section_header(f"{pollutant} — Reference vs MOx Sensor")
    with c_r:
        show_rolling = st.toggle("Trend line", value=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df.index, y=df[ref_col],
        name=f"{ref_col} (Ref)", opacity=0.35,
        line=dict(color=CHART_COLORS["ref"], width=1),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.2f}</b><extra>Reference</extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[sensor_col],
        name=f"{sensor_col} (Sensor)", opacity=0.25,
        line=dict(color=CHART_COLORS["sensor"], width=1),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.0f}</b><extra>Sensor</extra>",
    ), secondary_y=True)

    if show_rolling:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ref_col].rolling(f"{rolling_days}D").mean(),
            name=f"Ref {rolling_days}d avg",
            line=dict(color=CHART_COLORS["ref"], width=2.8),
            hovertemplate="%{x|%b %d %Y}<br><b>%{y:.2f}</b><extra>Ref trend</extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sensor_col].rolling(f"{rolling_days}D").mean(),
            name=f"Sensor {rolling_days}d avg",
            line=dict(color=CHART_COLORS["sensor"], width=2.8, dash="dot"),
            hovertemplate="%{x|%b %d %Y}<br><b>%{y:.0f}</b><extra>Sensor trend</extra>",
        ), secondary_y=True)

    fig.update_yaxes(title_text=f"{ref_col} [mg/m³]", secondary_y=False,
                     title_font=dict(color=CHART_COLORS["ref"]),
                     gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(title_text=f"{sensor_col} [nominal]", secondary_y=True,
                     title_font=dict(color=CHART_COLORS["sensor"]),
                     gridcolor=BORDER, linecolor=BORDER)
    apply_layout(fig, height=430, legend=dict(orientation="h", y=-0.22, x=0,
                 bgcolor="rgba(26,31,46,0.95)", bordercolor=BORDER, borderwidth=1))
    st.plotly_chart(fig, use_container_width=True)

    section_header("Monthly Distribution")
    df_tmp = df[[ref_col]].copy()
    df_tmp["Month"] = df_tmp.index.strftime("%b %Y")
    month_order = sorted(df_tmp["Month"].unique(), key=lambda x: pd.to_datetime(x))
    fig2 = px.box(df_tmp, x="Month", y=ref_col,
                  category_orders={"Month": month_order},
                  color_discrete_sequence=[CHART_COLORS["ref"]],
                  labels={ref_col: f"{ref_col} [mg/m³]"})
    fig2.update_traces(marker=dict(size=3, opacity=0.5), line=dict(width=1.5))
    apply_layout(fig2, height=360)
    fig2.update_xaxes(tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — H1: SENSOR ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    hypo_card(
        "C6H6 &amp; CO sensors achieve Pearson r &gt; 0.80 against certified references. "
        "NOx &amp; NO2 sensors fall below r = 0.70 due to cross-sensitivity and secondary atmospheric chemistry.",
        "H1 — Hypothesis",
    )
    with st.spinner("Computing correlations & OLS fits…"):
        h1 = compute_h1_correlations(df)

    section_header("Pearson r — Reference vs Sensor")
    cols4 = st.columns(4)
    for col_ui, (name, res) in zip(cols4, h1.items()):
        r = res["r"]
        a = abs(r)
        tier, tcls = (("High accuracy", "badge-high") if a > 0.80
                      else ("Medium", "badge-medium") if a > 0.70
                      else ("Low accuracy", "badge-low"))
        col_ui.markdown(f"""
        <div class="r-card">
            <div class="r-name">{name}</div>
            <div class="r-val">{r:+.3f}</div>
            <span class="badge {tcls}">{tier}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Regression Scatter Plot")
    c1, c2 = st.columns([1, 3])
    with c1:
        sel_name = st.radio("Pollutant", list(h1.keys()))
    with c2:
        res = h1[sel_name]
        subset, ref_c, sen_c, r = res["data"], res["ref"], res["sensor"], res["r"]
        ols = res["ols"]
        x_line = np.linspace(subset[sen_c].min(), subset[sen_c].max(), 300)
        y_line = ols.params.iloc[0] + ols.params.iloc[1] * x_line
        lc = SUCCESS if abs(r) > 0.80 else (WARNING if abs(r) > 0.70 else DANGER)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=subset[sen_c], y=subset[ref_c], mode="markers",
            marker=dict(color=CHART_COLORS["ref"], opacity=0.15, size=4),
            name="Observations",
            hovertemplate="Sensor: %{x:.0f}<br>Reference: %{y:.2f}<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=lc, width=2.8),
            name=f"OLS  r = {r:+.3f}",
        ))
        apply_layout(fig3, height=400,
                     xaxis_title=f"{sen_c} [nominal]",
                     yaxis_title=f"{ref_c} [mg/m³]",
                     title=f"<b>{sel_name}</b> — Reference vs Sensor")
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 OLS Regression Summary"):
        st.dataframe(res["ols"].summary2().tables[1].reset_index()
                     .rename(columns={"index": "Term"}).round(4),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — H2: CONFOUNDERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    hypo_card(
        f"Sensor accuracy degrades under high humidity (RH &gt; {rh_thresh:.0f}%) and high "
        f"temperature (T &gt; {t_thresh:.0f}°C), evidenced by significant OLS interaction terms "
        "and lower stratified Pearson r.",
        "H2 — Hypothesis",
    )
    with st.spinner("Computing stratified correlations…"):
        strat = compute_h2_stratified(df, t_thresh=t_thresh, rh_thresh=rh_thresh)

    section_header("Pearson r by Temperature × Humidity Stratum")
    st.caption(f"T threshold = {t_thresh}°C · RH threshold = {rh_thresh}%  "
               "— each cell is Pearson r within that environmental stratum")

    hcols = st.columns(4)
    for col_ui, name in zip(hcols, SENSOR_PAIRS.keys()):
        sub   = strat[strat["Pollutant"] == name]
        pivot = sub.pivot(index="T_bin", columns="RH_bin", values="r")
        fh = px.imshow(pivot, text_auto=".3f",
                       color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="equal")
        fh.update_traces(hovertemplate="T: %{y}<br>RH: %{x}<br><b>r = %{z:.3f}</b><extra></extra>")
        fh.update_layout(height=260, paper_bgcolor=CARD, plot_bgcolor=CARD,
                         font=dict(color=TEXT), margin=dict(t=44, b=8, l=8, r=8),
                         coloraxis_showscale=False,
                         title=dict(text=f"<b>{name}</b>", x=0.5, font=dict(size=13, color=TEXT)))
        col_ui.plotly_chart(fh, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("OLS Interaction Coefficients")
    st.caption("`T_x_sensor` and `RH_x_sensor` show how temperature and humidity modulate sensor response")

    with st.spinner("Fitting interaction models…"):
        h2_ols = compute_h2_ols(df)

    sel_h2 = st.selectbox("Pollutant", list(h2_ols.keys()), key="h2_sel")
    st.dataframe(h2_ols[sel_h2].summary2().tables[1].reset_index()
                 .rename(columns={"index": "Term"}).round(4),
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — H3: TEMPORAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    hypo_card(
        "Concentrations peak during morning (07–09 h) and evening (17–19 h) rush hours "
        "and are significantly higher in winter — confirmed by one-way ANOVA (p &lt; 0.05).",
        "H3 — Hypothesis",
    )
    with st.spinner("Computing temporal aggregations & ANOVA…"):
        h3 = compute_h3_temporal(df)

    plot_cols  = ["CO(GT)", "NOx(GT)", "NO2(GT)"]
    line_clrs  = [CHART_COLORS["co"], CHART_COLORS["nox"], CHART_COLORS["no2"]]
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    # ANOVA table
    section_header("One-Way ANOVA Results")
    rows_html = ""
    for grouping, d in [("Hour-of-day", h3["anova_hour"]), ("Month", h3["anova_month"])]:
        for col_name, res in d.items():
            sig_cls  = "sig"   if res["p"] < 0.05 else "insig"
            sig_text = "✅ Yes" if res["p"] < 0.05 else "❌ No"
            rows_html += (f"<tr><td>{grouping}</td><td>{col_name}</td>"
                          f"<td>{res['F']:.2f}</td><td>{res['p']:.2e}</td>"
                          f"<td class='{sig_cls}'>{sig_text}</td></tr>")
    st.markdown(f"""
    <table class="anova-table">
      <thead><tr><th>Grouping</th><th>Pollutant</th><th>F-stat</th><th>p-value</th><th>p &lt; 0.05?</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table><br>""", unsafe_allow_html=True)

    # Diurnal
    section_header("Diurnal Pattern — Hourly Mean ± 95% CI")
    hourly, hourly_sem = h3["hourly_mean"], h3["hourly_sem"]
    fig_d = go.Figure()
    for col_name, c in zip(plot_cols, line_clrs):
        ci = 1.96 * hourly_sem[col_name]
        fig_d.add_trace(go.Scatter(
            x=list(hourly.index) + list(hourly.index[::-1]),
            y=list(hourly[col_name] + ci) + list((hourly[col_name] - ci)[::-1]),
            fill="toself", fillcolor=c, opacity=0.12,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_d.add_trace(go.Scatter(
            x=hourly.index, y=hourly[col_name],
            mode="lines+markers", name=col_name,
            line=dict(color=c, width=2.2), marker=dict(size=5),
            hovertemplate=f"Hour %{{x}}:00<br><b>%{{y:.3f}}</b><extra>{col_name}</extra>",
        ))
    for x0, x1, label in [(7, 9, "Morning rush"), (17, 19, "Evening rush")]:
        fig_d.add_vrect(x0=x0, x1=x1, fillcolor=WARNING, opacity=0.07, line_width=0,
                        annotation_text=label, annotation_position="top left",
                        annotation=dict(font_size=10, font_color=WARNING))
    apply_layout(fig_d, height=390, xaxis_title="Hour of Day",
                 xaxis=dict(tickmode="linear", tick0=0, dtick=2, gridcolor=BORDER))
    st.plotly_chart(fig_d, use_container_width=True)

    # Seasonal bar
    section_header("Seasonal Pattern — Monthly Mean Concentration")
    monthly = h3["monthly_mean"].copy()
    monthly.index = [month_names.get(m, m) for m in monthly.index]
    monthly.index.name = "month"
    mdf = monthly[plot_cols].reset_index()
    fig_s = px.bar(mdf, x="month", y=plot_cols, barmode="group",
                   color_discrete_sequence=line_clrs,
                   labels={"month": "Month", "value": "Mean concentration", "variable": "Pollutant"})
    fig_s.update_traces(marker_line_width=0)
    apply_layout(fig_s, height=360)
    st.plotly_chart(fig_s, use_container_width=True)

    # Heatmap
    section_header("CO(GT) Mean — Hour × Month Heatmap")
    hm = h3["heatmap"].copy()
    hm.columns = [month_names.get(m, m) for m in hm.columns]
    fig_hm = px.imshow(hm, text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto",
                       labels=dict(color="CO(GT) [mg/m³]", x="Month", y="Hour"))
    fig_hm.update_traces(
        hovertemplate="Hour %{y}:00 · %{x}<br><b>CO = %{z:.2f} mg/m³</b><extra></extra>")
    apply_layout(fig_hm, height=460)
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — H4: PCA
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    hypo_card(
        "CO, NOx, and C6H6 share a common traffic-emission origin (r &gt; 0.60) and cluster on "
        "the same principal component. NO₂, as a secondary atmospheric pollutant, loads on a separate axis.",
        "H4 — Hypothesis",
    )
    with st.spinner("Running PCA…"):
        h4 = compute_h4_pca(df)

    loadings, scores, ev = h4["loadings"], h4["scores"], h4["explained"] * 100

    col_c, col_d = st.columns(2)
    with col_c:
        section_header("Correlation Matrix")
        fig_corr = px.imshow(h4["corr"], text_auto=".3f",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="equal")
        fig_corr.update_traces(
            hovertemplate="%{x} × %{y}<br><b>r = %{z:.3f}</b><extra></extra>")
        apply_layout(fig_corr, height=370)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_d:
        section_header("Scree Plot")
        cum_ev = np.cumsum(ev)
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(4)], y=ev,
            marker_color=PRIMARY, marker_line_width=0, name="Component variance",
            hovertemplate="<b>PC%{x}</b><br>%{y:.1f}% variance<extra></extra>",
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(4)], y=cum_ev,
            mode="lines+markers", name="Cumulative",
            line=dict(color=DANGER, width=2.2), marker=dict(size=7, color=DANGER),
            hovertemplate="<b>PC%{x}</b><br>%{y:.1f}% cumulative<extra></extra>",
        ))
        apply_layout(fig_scree, height=370, yaxis_title="Explained Variance (%)")
        st.plotly_chart(fig_scree, use_container_width=True)

    section_header("PCA Biplot — Loading Vectors")
    st.caption("Arrow direction = how each pollutant contributes to PCs. "
               "Pollutants with similar arrows are correlated.")

    pca_colors = [DANGER, PRIMARY, SUCCESS, "#C084FC"]
    scale = 3.0
    fig_bi = go.Figure()
    fig_bi.add_trace(go.Scatter(
        x=scores[:, 0], y=scores[:, 1], mode="markers",
        marker=dict(color=MUTED, opacity=0.15, size=4),
        name="Observations", hoverinfo="skip",
    ))
    for col_name, c in zip(REF_COLS, pca_colors):
        lx = loadings.loc[col_name, "PC1"] * scale
        ly = loadings.loc[col_name, "PC2"] * scale
        fig_bi.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly], mode="lines",
            line=dict(color=c, width=2.5), showlegend=False, hoverinfo="skip",
        ))
        fig_bi.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers+text",
            marker=dict(size=10, color=c, symbol="circle"),
            text=[f"<b>{col_name}</b>"], textposition="top center",
            textfont=dict(color=c, size=12), name=col_name,
            hovertemplate=(f"<b>{col_name}</b><br>"
                           f"PC1: {loadings.loc[col_name,'PC1']:.3f}<br>"
                           f"PC2: {loadings.loc[col_name,'PC2']:.3f}<extra></extra>"),
        ))
    fig_bi.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))
    fig_bi.add_vline(x=0, line=dict(color=BORDER, width=1, dash="dot"))
    apply_layout(fig_bi, height=500,
                 xaxis_title=f"PC1  ({ev[0]:.1f}% variance)",
                 yaxis_title=f"PC2  ({ev[1]:.1f}% variance)")
    st.plotly_chart(fig_bi, use_container_width=True)

    with st.expander("📋 Component Loadings Table"):
        st.dataframe(loadings.round(3), use_container_width=True)
