# app.py — BESS-Arbitrage (einfach): SoC + Netzanschluss + Sperr-Slots
#
# Tabelleneingang (Excel/CSV) — minimal erforderliche Spalten:
#   • Zeit (15‑Min)
#   • SoC_kWh  (Baseline-SoC je Slot; unsere Arbitrage darf NICHT darunter fallen)
#   • Netzleistung (freier Netzanschluss-Headroom je Slot; symmetrisch für Laden/Entladen)
#   • BESS (0/1)  → 1 bedeutet: Slot belegt, also kein zusätzlicher Handel
#
# Preisdatei (Excel/CSV): Zeit, Preis (€/MWh | €/kWh | ct/kWh)
#
# Optimierung (Szenario A/B):
#   • Entscheidet zusätzliche Lade-/Entladeleistungen (kW) und extra-SoC ≥ 0 oberhalb Baseline-SoC
#   • Nebenbedingungen pro Slot:
#       - 0 ≤ p_ch ≤ min(P_max, Netzleistung) * (1 − busy)
#       - 0 ≤ p_dis ≤ min(P_max, Netzleistung) * (1 − busy)
#       - 0 ≤ soc_extra ≤ E_max − SoC_baseline
#       - SoC-Dynamik: soc_extra[t] = soc_extra[t−1] + (η_ch·p_ch − p_dis/η_dis)·Δt
#   • Optional: Jahres‑Zyklenlimit (nur zusätzliche Entladung):  Σ_year(p_dis·Δt)/E_max ≤ Cap
#   • Optional: Energie-Bilanz über den Horizont (Start‑SoC_extra = End‑SoC_extra)
#   • Zielfunktion: ∑ ((Preis−Fee_sell)·E_dis − (Preis+Fee_buy)·E_ch)/1000
#
# Szenarien:
#   A  … mit Sperren (busy aus Datei)
#   B  … frei (busy=0), sonst identische Grenzen

import math
from io import BytesIO
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

try:
    import pulp
except Exception:
    pulp = None

st.set_page_config(page_title="BESS – Arbitrage (SoC + Netzanschluss)", layout="wide")
st.title("⚡ BESS-Arbitrage mit SoC, Netzanschluss & Sperr-Slots")

# ---------------------- Uploads ----------------------
cu1, cu2 = st.columns(2)
with cu1:
    base_file = st.file_uploader("Basistabelle (Zeit, SoC_kWh, Netzleistung, BESS)", type=["xlsx","xls","csv"], key="base")
with cu2:
    price_file = st.file_uploader("Day-Ahead-Preise (Zeit, Preis)", type=["xlsx","xls","csv"], key="price")

if base_file is None or price_file is None:
    st.info("Bitte beide Dateien laden.")
    st.stop()

# ---------------------- Read helpers ----------------------
def read_any(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f, sheet_name=0)

base_raw = read_any(base_file)
price_raw = read_any(price_file)

# ---------------------- Column mapping ----------------------
st.subheader("Spaltenzuordnung")
base_cols  = list(base_raw.columns)
price_cols = list(price_raw.columns)

def guess(colnames, keys):
    low = {c.lower(): c for c in colnames}
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return colnames[0]

b_time = st.selectbox("Basistabelle: Zeitstempel", base_cols, index=base_cols.index(guess(base_cols,["zeit","time","date"])) )
b_soc  = st.selectbox("Basistabelle: SoC_kWh (Baseline)", base_cols, index=base_cols.index(guess(base_cols,["soc","ladestand"])) )
b_net  = st.selectbox("Basistabelle: Netzleistung (Headroom)", base_cols, index=base_cols.index(guess(base_cols,["netz","head","anschluss","leistung"])) )
b_busy = st.selectbox("Basistabelle: BESS-Sperre (0/1)", base_cols, index=base_cols.index(guess(base_cols,["bess","slot","sperre","busy","block"])) )

p_time = st.selectbox("Preise: Zeitstempel", price_cols, index=price_cols.index(guess(price_cols,["zeit","time","date"])) )
p_val  = st.selectbox("Preise: Preis-Spalte", price_cols, index=price_cols.index(guess(price_cols,["preis","price","eur","ct"])) )

price_unit = st.radio("Preiseinheit", ["€/MWh","€/kWh","ct/kWh"], horizontal=True)
net_unit   = st.radio("Einheit der Netzleistung", ["kW","W"], horizontal=True)

# ---------------------- Batterie-Parameter ----------------------
st.subheader("Batterie- & Handelsparameter")
cp1, cp2, cp3 = st.columns(3)
with cp1:
    E_max = st.number_input("E_max nutzbar [kWh]", min_value=1.0, value=150.0, step=1.0)
with cp2:
    P_max = st.number_input("P_max (sym) [kW]", min_value=0.1, value=100.0, step=1.0)
with cp3:
    rte_pct = st.number_input("RTE [%]", min_value=1.0, max_value=100.0, value=95.0, step=0.5)

cp4, cp5, cp6 = st.columns(3)
with cp4:
    soc0_extra_pct = st.slider("Start-SoC_extra [% von E_max]", min_value=0, max_value=100, value=0)
with cp5:
    fix_final = st.checkbox("Finaler SoC_extra = Start-SoC_extra", value=True)
with cp6:
    cycles_cap = st.number_input("Max. Vollzyklen/Jahr (nur Zusatz) [#]", min_value=0.0, value=0.0, step=0.5, help="0 = keine Begrenzung")

cp7, cp8, cp9 = st.columns(3)
with cp7:
    fees = st.number_input("Gebühren [€/MWh] (Kauf/Verkauf)", min_value=0.0, value=0.0, step=0.1)
with cp8:
    P_grid_max = st.number_input("Zusätzl. Grid-Cap [kW] (0 = ignorieren)", min_value=0.0, value=0.0, step=10.0)
with cp9:
    st.write("")

eta_rt  = rte_pct/100.0
eta_ch  = eta_rt ** 0.5
eta_dis = eta_rt ** 0.5
soc0_extra_kwh = E_max * (soc0_extra_pct/100.0)

# ---------------------- Datenaufbereitung ----------------------
base = base_raw[[b_time, b_soc, b_net, b_busy]].copy()
base.columns = ["ts","soc_base_kwh","net_headroom","busy"]
base["ts"] = pd.to_datetime(base["ts"], errors="coerce")
base = base.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

# Einheiten
base["soc_base_kwh"] = pd.to_numeric(base["soc_base_kwh"], errors="coerce").fillna(0.0)
if net_unit == "W":
    base["net_headroom"] = pd.to_numeric(base["net_headroom"], errors="coerce").fillna(0.0)/1000.0
else:
    base["net_headroom"] = pd.to_numeric(base["net_headroom"], errors="coerce").fillna(0.0)
base["net_headroom"] = base["net_headroom"].clip(lower=0.0)

b = pd.to_numeric(base["busy"], errors="coerce").fillna(0)
base["busy"] = (b > 0).astype(int)

price = price_raw[[p_time, p_val]].copy()
price.columns = ["ts","price_raw"]
price["ts"] = pd.to_datetime(price["ts"], errors="coerce")
price = price.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

if price_unit == "€/MWh":
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce")
elif price_unit == "€/kWh":
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 1000.0
else:
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 10.0

DF = pd.merge_asof(base.sort_values("ts"), price.sort_values("ts"), on="ts", direction="nearest", tolerance=pd.Timedelta("8min"))
if DF["price_eur_per_mwh"].isna().any():
    st.error("Preiswerte konnten nicht gematcht werden. Bitte Zeitraster/Zeitzone prüfen.")
    st.stop()

# Zeitschritt
step = DF["ts"].diff().median()
if pd.isna(step) or step == pd.Timedelta(0):
    st.error("Konnte die Zeitschrittweite nicht bestimmen.")
    st.stop()

dt_h = step.total_seconds()/3600.0
fee_buy = float(fees)
fee_sell = float(fees)

# Effektive Grid-Cap anwenden (optional)
if P_grid_max and P_grid_max > 0:
    DF["net_eff"] = np.minimum(DF["net_headroom"].values, P_grid_max)
else:
    DF["net_eff"] = DF["net_headroom"].values

# ---------------------- Optimierer ----------------------
def optimize_with_soc(df: pd.DataFrame,
                      E_max: float, P_max: float,
                      eta_ch: float, eta_dis: float,
                      dt_h: float,
                      soc0_extra_kwh: float,
                      fix_final: bool,
                      fee_buy: float, fee_sell: float,
                      cycles_cap: float) -> Optional[pd.DataFrame]:
    if pulp is None:
        st.error("PuLP fehlt: pip install pulp (und ggf. coinor-cbc)")
        return None

    n = len(df)
    m = pulp.LpProblem("BESS_SOC_NET", pulp.LpMaximize)

    ch  = pulp.LpVariable.dicts("ch_kw",  range(n), lowBound=0)
    dis = pulp.LpVariable.dicts("dis_kw", range(n), lowBound=0)
    soc = pulp.LpVariable.dicts("soc_extra_kwh", range(n), lowBound=0)

    # Ziel
    m += pulp.lpSum( ((float(df.loc[i,"price_eur_per_mwh"]) - fee_sell) * dis[i]
                      - (float(df.loc[i,"price_eur_per_mwh"]) + fee_buy) * ch[i]) * (dt_h/1000.0)
                    for i in range(n) )

    # Dynamik
    m += soc[0] == soc0_extra_kwh + (eta_ch*ch[0] - (dis[0]/eta_dis)) * dt_h
    for i in range(1, n):
        m += soc[i] == soc[i-1] + (eta_ch*ch[i] - (dis[i]/eta_dis)) * dt_h

    # SoC-Headroom gegen Baseline
    for i in range(n):
        headroom_soc = max(0.0, E_max - float(df.loc[i, "soc_base_kwh"]))
        m += soc[i] <= headroom_soc

    # Leistungslimits je Slot (Busy + Netz + P_max)
    for i in range(n):
        busy = int(df.loc[i, "busy"])  # 0/1
        net_cap = float(df.loc[i, "net_eff"])  # kW
        cap = min(P_max, max(0.0, net_cap))
        m += ch[i]  <= cap * (1 - busy)
        m += dis[i] <= cap * (1 - busy)

    # Jahres-Zyklenlimit (nur Zusatz-Entladung)
    if cycles_cap and cycles_cap > 0 and E_max > 0:
        years = df["ts"].dt.year.to_list()
        from collections import defaultdict
        idx_by_year = defaultdict(list)
        for i, y in enumerate(years):
            idx_by_year[y].append(i)
        for y, idxs in idx_by_year.items():
            m += pulp.lpSum(dis[i] for i in idxs) * dt_h <= cycles_cap * E_max

    if fix_final:
        m += soc[n-1] == soc0_extra_kwh

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        st.error(f"Optimierung nicht optimal: {pulp.LpStatus[m.status]}")
        return None

    out = df[["ts","soc_base_kwh","net_eff","busy","price_eur_per_mwh"]].copy()
    out["p_ch_kw"]  = [pulp.value(ch[i]) for i in range(n)]
    out["p_dis_kw"] = [pulp.value(dis[i]) for i in range(n)]
    out["soc_extra_kwh"] = [pulp.value(soc[i]) for i in range(n)]

    out["soc_total_kwh"] = out["soc_base_kwh"] + out["soc_extra_kwh"]
    out["e_ch_kwh"]  = out["p_ch_kw"]  * dt_h
    out["e_dis_kwh"] = out["p_dis_kw"] * dt_h
    out["revenue_eur"] = ((out["price_eur_per_mwh"] - fee_sell) * out["e_dis_kwh"]
                           - (out["price_eur_per_mwh"] + fee_buy) * out["e_ch_kwh"]) / 1000.0
    return out

# Run
run = st.button("Optimierung starten", type="primary")
if not run:
    st.stop()

# A: mit Sperren
res_A = optimize_with_soc(DF.copy(), E_max, P_max, eta_ch, eta_dis, dt_h,
                          soc0_extra_kwh, fix_final, fee_buy, fee_sell, cycles_cap)
# B: frei (busy=0)
DF_free = DF.copy()
DF_free["busy"] = 0
res_B = optimize_with_soc(DF_free, E_max, P_max, eta_ch, eta_dis, dt_h,
                          soc0_extra_kwh, fix_final, fee_buy, fee_sell, cycles_cap)

if res_A is None or res_B is None:
    st.stop()

# ---------------------- KPIs ----------------------
def kpis(df):
    rev = df["revenue_eur"].sum()
    zyk = df["e_dis_kwh"].sum()/E_max if E_max>0 else np.nan  # zusätzliche Vollzyklen (nur extra)
    return rev, zyk

revA, cycA = kpis(res_A)
revB, cycB = kpis(res_B)

c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Erlös A – mit Sperren [€]", f"{revA:,.0f}")
with c2:
    st.metric("Erlös B – frei [€]", f"{revB:,.0f}")
with c3:
    st.metric("Delta B − A [€]", f"{(revB-revA):,.0f}")

c4,c5 = st.columns(2)
with c4:
    st.metric("A: Zusatz‑Vollzyklen [#]", f"{cycA:,.1f}")
with c5:
    st.metric("B: Zusatz‑Vollzyklen [#]", f"{cycB:,.1f}")

# ---------------------- Plots (Ausschnitt) ----------------------
import altair as alt
sample = 7*24*int(round(1/dt_h))
sl = slice(0, min(sample, len(res_A)))

chart_price = alt.Chart(res_A.iloc[sl]).mark_line().encode(
    x="ts:T", y=alt.Y("price_eur_per_mwh:Q", title="Preis [€/MWh]")
).properties(height=200)

A_long = res_A.iloc[sl][["ts","p_ch_kw","p_dis_kw"]].melt("ts", var_name="Signal", value_name="kW")
chart_A = alt.Chart(A_long).mark_bar().encode(
    x="ts:T", y=alt.Y("kW:Q", title="Leistung A [kW]"), color="Signal:N"
).properties(height=200)

soc_view = res_A.iloc[sl][["ts","soc_base_kwh","soc_total_kwh"]].melt("ts", var_name="Signal", value_name="kWh")
chart_soc = alt.Chart(soc_view).mark_line().encode(
    x="ts:T", y=alt.Y("kWh:Q", title="SoC [kWh]"), color="Signal:N"
).properties(height=200)

st.altair_chart(chart_price, use_container_width=True)
st.altair_chart(chart_A, use_container_width=True)
st.altair_chart(chart_soc, use_container_width=True)

# ---------------------- Export ----------------------
out = DF[["ts","soc_base_kwh","net_eff","busy","price_eur_per_mwh"]].merge(
    res_A[["ts","p_ch_kw","p_dis_kw","soc_extra_kwh","soc_total_kwh","e_ch_kwh","e_dis_kwh","revenue_eur"]].rename(columns={
        "p_ch_kw":"p_ch_A_kw","p_dis_kw":"p_dis_A_kw","soc_extra_kwh":"soc_extra_A_kwh","soc_total_kwh":"soc_total_A_kwh",
        "e_ch_kwh":"e_ch_A_kwh","e_dis_kwh":"e_dis_A_kwh","revenue_eur":"revenue_A_eur"}), on="ts", how="left")\
    .merge(
    res_B[["ts","p_ch_kw","p_dis_kw","soc_extra_kwh","soc_total_kwh","e_ch_kwh","e_dis_kwh","revenue_eur"]].rename(columns={
        "p_ch_kw":"p_ch_B_kw","p_dis_kw":"p_dis_B_kw","soc_extra_kwh":"soc_extra_B_kwh","soc_total_kwh":"soc_total_B_kwh",
        "e_ch_kwh":"e_ch_B_kwh","e_dis_kwh":"e_dis_B_kwh","revenue_eur":"revenue_B_eur"}), on="ts", how="left")

kpi_df = pd.DataFrame({
    "Kennzahl": [
        "Erlös A [€]","Erlös B [€]","Delta B−A [€]",
        "A: Zusatz‑Vollzyklen [#]","B: Zusatz‑Vollzyklen [#]",
        "RTE [%]","P_max [kW]","E_max [kWh]","Grid-Cap [kW]"
    ],
    "Wert": [round(revA,2), round(revB,2), round(revB-revA,2), round(cycA,2), round(cycB,2), rte_pct, P_max, E_max, (P_grid_max or 0)]
})

bio = BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as w:
    out.to_excel(w, index=False, sheet_name="Zeitreihen")
    kpi_df.to_excel(w, index=False, sheet_name="KPIs")

bio.seek(0)

st.download_button(
    "Ergebnisse als Excel herunterladen",
    data=bio,
    file_name="BESS_Arbitrage_SoC_Grid.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
