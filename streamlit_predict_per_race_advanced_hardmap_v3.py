
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Tuple
from scipy.special import expit, logit

st.set_page_config(page_title="Greyhound Picks — Hard-Mapped (Fixed v3)", layout="wide")
st.title("🐾 Greyhound Picks — Hard-Mapped to Your Columns (v3, KeyError fix)")

st.markdown("""
**Exact columns expected:**  
`Date`, `#`, `Box`, `Dog Name`, `Form`, `Win %`, `Place %`, `Days Last Run`,  
`5th Last`, `4th Last`, `3rd Last`, `2ndLast`, `Last`, `Average(5)`, `Length Adv(5)`, `Prize Money Ind`,  
`Back`, `Lay`, `Winner`, `Rating`.

This version removes the `KeyError: early_proxy` by using a groupwise z-score that works on any Series (not just DataFrame columns).
""")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    commission = st.number_input("Commission (e.g., AU 6% = 0.06)", min_value=0.0, max_value=0.2, value=0.06, step=0.01)
    w_model = st.slider("Logit-pool weight for MODEL (rest = market)", 0.0, 1.0, 0.60, 0.05)
    gamma1 = st.slider("Logistic scale γ₁ (score→prob)", 0.2, 2.0, 1.2, 0.1)
    topk = st.selectbox("Max selections per race (per side)", [1,2,3], index=0)
    include_back = st.checkbox("Also compute Back score/picks", value=True)
    min_lay_odds, max_lay_odds = st.slider("Lay odds range filter", 1.01, 100.0, (1.5, 12.0), 0.01)
    min_back_odds, max_back_odds = st.slider("Back odds range filter", 1.01, 100.0, (1.5, 10.0), 0.01)

st.subheader("Upload CSV or XLSX (columns must match the list above)")
file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

def read_file(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

REQ_COLS = [
    "Date", "#", "Box", "Dog Name", "Form", "Win %", "Place %", "Days Last Run",
    "5th Last", "4th Last", "3rd Last", "2ndLast", "Last", "Average(5)",
    "Length Adv(5)", "Prize Money Ind", "Back", "Lay", "Winner", "Rating"
]

if file is None:
    st.info("Upload a file to run selections.")
    st.stop()

df_raw = read_file(file)
missing = [c for c in REQ_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = df_raw.copy()

# ---------- Coercions (hard-map) ----------
def to_float(s: pd.Series) -> pd.Series:
    # convert '-', ' ' to NaN then numeric
    s2 = pd.to_numeric(s.replace({"-": np.nan, "": np.nan}), errors="coerce")
    return s2.astype(float)

for c in ["Back","Lay","Average(5)","Length Adv(5)","Prize Money Ind","5th Last","4th Last","3rd Last","2ndLast","Last"]:
    df[c] = to_float(df[c])

for c in ["Win %","Place %"]:
    df[c] = to_float(df[c]) / 100.0  # convert from % to [0,1]

df["Days Last Run"] = to_float(df["Days Last Run"])
df["Box"] = to_float(df["Box"]).fillna(1.0)

# ---------- Race grouping ----------
# If no explicit race grouping, treat as single race
df["RaceId"] = "RACE_1"

# ---------- Helpers ----------
def z_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Z-score a Series by group labels; safe for non-column Series."""
    s = pd.Series(values.values if isinstance(values, pd.Series) else values, index=df.index, dtype=float)
    mu = s.groupby(groups).transform("mean")
    sd = s.groupby(groups).transform("std").replace(0, 1.0)
    return (s - mu) / sd

# ---------- Feature engineering ----------
Avg5 = df["Average(5)"].fillna(df["Average(5)"].median())
Z_Avg5 = z_by_group(Avg5, df["RaceId"])

# Early speed proxy (no column provided). Positive means **slowing** relative to earlier run.
early_proxy = (to_float(df["2ndLast"]) + to_float(df["Last"])) / 2.0 - to_float(df["5th Last"])
early_proxy = early_proxy.fillna(early_proxy.median())
Z_Early = -z_by_group(early_proxy, df["RaceId"])  # invert so higher = slower early

Z_Days = z_by_group(df["Days Last Run"].fillna(df["Days Last Run"].median()), df["RaceId"])

# Drift not present — set to 0
Z_Drift = pd.Series(0.0, index=df.index)

# Slowing score from linear trend across last 5 runs
vals = ["5th Last","4th Last","3rd Last","2ndLast","Last"]
X = np.arange(len(vals))
def slope_row(row) -> float:
    y = row[vals].astype(float).values
    if np.isnan(y).all():
        return 0.0
    m = ~np.isnan(y)
    if m.sum() < 2:
        return 0.0
    x = X[m]
    yy = y[m]
    xbar = x.mean()
    ybar = yy.mean()
    num = ((x - xbar)*(yy - ybar)).sum()
    den = ((x - xbar)**2).sum()
    if den == 0:
        return 0.0
    return float(num/den)

Slow_raw = df[vals].apply(slope_row, axis=1)
Slow = z_by_group(Slow_raw, df["RaceId"]).clip(-3, 3)

# Box penalty scaled to [0,1] across file
box = df["Box"].fillna(df["Box"].median())
if float(box.max()) > float(box.min()):
    BoxPenalty = (box - box.min()) / (box.max() - box.min())
else:
    BoxPenalty = pd.Series(0.0, index=df.index)

WinPct = df["Win %"].fillna(0.0)
PlacePct = df["Place %"].fillna(0.0)

# ---------- LayScore ----------
LayScore = (
    -0.45*WinPct
    -0.20*PlacePct
    +0.22*Z_Avg5
    +0.18*Z_Early
    +0.12*BoxPenalty
    +0.10*Z_Days
    +0.10*Slow
    +0.08*Z_Drift
    +0.10*BoxPenalty*Z_Early
)

# Normalize within race and convert to model probabilities
Sstar = z_by_group(LayScore, df["RaceId"])
p_lose_model = expit(0 + gamma1*Sstar)
p_win_model = 1.0 - p_lose_model

# Market probability from mid
Mid = np.where(~df["Back"].isna() & ~df["Lay"].isna(), (df["Back"] + df["Lay"]) / 2.0, df["Lay"].fillna(df["Back"]))
Mid = np.clip(Mid, 1.01, None)
p_mkt_raw = 1.0 / Mid
p_mkt = np.minimum(1.0, p_mkt_raw * (1 + commission))

def safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return logit(p)

p_win_true = expit(w_model*safe_logit(p_win_model) + (1.0 - w_model)*safe_logit(p_mkt))
p_lose_true = 1.0 - p_win_true

# EV per $1 liability and threshold
L = np.clip(df["Lay"].astype(float).values, 1.01, None)
EV_liab = (1 - p_win_true)*(1 - commission)/np.maximum(L - 1.0, 1e-6) - p_win_true
p_win_thresh = (1 - commission) / (L - commission)

# Optional Back score (simple mirror)
BackScore = (
     0.45*WinPct
    +0.20*PlacePct
    -0.20*Z_Avg5
    -0.18*Z_Early
    -0.12*BoxPenalty
    -0.10*Z_Days
    -0.05*Slow
)

# ---------- Assemble & filter ----------
base = pd.DataFrame({
    "RaceId": df["RaceId"],
    "Date": df["Date"],
    "#": df["#"],
    "Box": df["Box"],
    "Dog Name": df["Dog Name"],
    "Back": df["Back"],
    "Lay": df["Lay"],
    "Win %": WinPct,
    "Place %": PlacePct,
    "Average(5)": df["Average(5)"],
    "LayScore": LayScore,
    "Sstar": Sstar,
    "p_win_model": p_win_model,
    "p_win_mkt": p_mkt,
    "p_win_true": p_win_true,
    "EV_per_$1_liab": EV_liab,
    "p_win_thresh": p_win_thresh,
    "is_plus_EV": p_win_true < p_win_thresh,
    "BackScore": BackScore if include_back else np.nan,
})

mask = (base["Lay"].between(min_lay_odds, max_lay_odds)) & (base["Back"].between(min_back_odds, max_back_odds) | base["Back"].isna())
base_f = base[mask].copy()

def topk_per_race(df_in: pd.DataFrame, by_col: str, ascending: bool, k: int) -> pd.DataFrame:
    df_in = df_in.copy()
    df_in["_rank"] = df_in.groupby("RaceId")[by_col].rank(method="first", ascending=ascending)
    out = df_in[df_in["_rank"] <= k].drop(columns=["_rank"])
    return out.sort_values(["RaceId", by_col], ascending=[True, ascending])

lay_picks = topk_per_race(base_f, "LayScore", ascending=False, k=topk)
if include_back:
    back_picks = topk_per_race(base_f, "BackScore", ascending=False, k=topk)
else:
    back_picks = pd.DataFrame(columns=base_f.columns)

combined = lay_picks.assign(PickType="LAY")
if not back_picks.empty:
    combined = pd.concat([combined, back_picks.assign(PickType="BACK")], ignore_index=True)

# ---------- Display ----------
st.subheader("Summary")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total races", int(base["RaceId"].nunique()))
with c2:
    st.metric("Lay selections", int(lay_picks.shape[0]))
with c3:
    st.metric("Plus-EV lays", int(lay_picks["is_plus_EV"].sum()) if not lay_picks.empty else 0)

st.subheader("Lay picks")
st.dataframe(lay_picks, use_container_width=True)
st.download_button("⬇️ Download Lay picks CSV", df_to_csv_bytes(lay_picks), "per_race_lay_picks.csv")

if include_back:
    st.subheader("Back picks")
    st.dataframe(back_picks, use_container_width=True)
    st.download_button("⬇️ Download Back picks CSV", df_to_csv_bytes(back_picks), "per_race_back_picks.csv")

st.subheader("Combined picks")
st.dataframe(combined, use_container_width=True)
st.download_button("⬇️ Download Combined picks CSV", df_to_csv_bytes(combined), "per_race_picks_combined.csv")

st.caption("Hard-mapped to your columns (v3). Uses groupwise z-scores that accept plain Series — no column naming required.")
