# app.py — Streamlit-App zur Ermittlung von Batterieerlösen
# Szenario A: Arbitrage zusätzlich zur Eigenverbrauchslogik (mit Einschränkungen aus Basistabelle)
# Szenario B: Freies Arbitrage-Handeln ohne Einschränkungen (nur E_max/P_max/RTE)
#
# Erwartung an die Basistabelle (Beispiel aus Ihrer Datei):
# - Zeitstempelspalte (z. B. "Zeit") in 15-min-Auflösung
# - Baseline-SoC in kWh (z. B. "SoC_kWh")
# - Baseline-Ladeenergie in kWh pro 15 min (z. B. "ch_pv_kWh")
# - Baseline-Entladeenergie in kWh pro 15 min (z. B. "dh_ev_kWh")
#   (Diese bilden die Eigenverbrauchslogik ab. Wir rechnen obendrauf eine Arbitrage-Optimierung.)
#
# Erwartung an die Preisdatei:
# - Zeitstempelspalte
# - Preisspalte (Einheit wählbar: €/MWh, €/kWh, ct/kWh)
#
# Installationshinweis (falls nötig):
#   pip install streamlit pulp pandas numpy altair openpyxl

import math
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- Optional: lineare Optimierung mit PuLP ---
try:
    import pulp
except Exception as e:
    pulp = None

st.set_page_config(page_title="BESS Arbitrage – Einschränkungen vs. frei", layout="wide")
st.title("⚡ BESS-Arbitrage: Eigenverbrauch + Märkte vs. frei handeln")

st.markdown(
    """
Diese App vergleicht zwei Fälle:

**A) Mit Einschränkungen:** Ihre Basistabelle (Eigenverbrauch) bleibt unberührt. Die Batterie darf zusätzlich marktgetrieben handeln, **ohne** die Baseline zu stören. Dazu werden folgende Restriktionen aus der Basistabelle abgeleitet:
- SoC darf **zu keinem Zeitpunkt** unter dem Baseline-SoC fallen (wir stehlen keine Energie, die für EV/PV benötigt wird).
- Totale Lade-/Entladeleistung je 15-min-Intervall darf die eingestellte **P_max** nicht überschreiten. Der **Baseline-Anteil** belegt einen Teil davon, der **Arbitrage-Anteil** nutzt die Restleistung.

**B) Frei handeln:** Klassische Arbitrage-Optimierung ohne Baseline-Zwänge. Nur E_max/P_max/RTE und SoC-Randbedingungen gelten.
"""
)

# ---------------------- Datei-Uploads ----------------------
col_up1, col_up2 = st.columns([1,1])
with col_up1:
    base_file = st.file_uploader("Basistabelle (Eigenverbrauch) – Excel/CSV", type=["xlsx", "xls", "csv"], key="base")
with col_up2:
    price_file = st.file_uploader("Day-Ahead-Preise – Excel/CSV", type=["xlsx", "xls", "csv"], key="price")

if base_file is None or price_file is None:
    st.info("Bitte beide Dateien laden (Basistabelle und Preise). Beispiel: Ihre Datei 'BESS_Single_Eigenverbrauch_*.xlsx' und eine separate Day-Ahead-Preisdatei in 15-min-Auflösung.")
    st.stop()

# ---------------------- Einlese-Helper ----------------------
def read_any(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        # Excel: wähle erstes Sheet automatisch
        df = pd.read_excel(file, sheet_name=0)
    return df

base_df_raw = read_any(base_file)
price_df_raw = read_any(price_file)

# ---------------------- Spaltenzuordnung ----------------------
st.subheader("Spaltenzuordnung")

# Kandidaten vorfüllen
base_cols = list(base_df_raw.columns)
price_cols = list(price_df_raw.columns)

# Heuristiken für Vorauswahl
def guess(colnames, keys):
    low = {c.lower(): c for c in colnames}
    for k in keys:
        for c_low, c in low.items():
            if k in c_low:
                return c
    return None

base_time_guess = guess(base_cols, ["zeit", "time", "timestamp", "datetime", "date"])
base_soc_guess = guess(base_cols, ["soc_kwh", "soc", "state of charge", "ladestand", "ladung"])
base_ch_guess = guess(base_cols, ["ch_", "charge_kwh", "ladung_kwh", "ladeenergie"])
base_dis_guess = guess(base_cols, ["dh_", "dis", "entladung_kwh", "discharge_kwh"])  # dh_ev_kWh in Ihrer Datei

price_time_guess = guess(price_cols, ["zeit", "time", "timestamp", "datetime", "date"]) 
price_val_guess  = guess(price_cols, ["preis", "price", "eur", "ct", "€/mwh", "euro"]) 

c1, c2, c3, c4 = st.columns(4)
with c1:
    base_time_col = st.selectbox("Basistabelle: Zeitstempel", base_cols, index=(base_cols.index(base_time_guess) if base_time_guess in base_cols else 0))
with c2:
    base_soc_col = st.selectbox("Basistabelle: Baseline-SoC [kWh]", base_cols, index=(base_cols.index(base_soc_guess) if base_soc_guess in base_cols else 0))
with c3:
    base_ch_col = st.selectbox("Basistabelle: Baseline-Ladeenergie je 15 min [kWh]", base_cols, index=(base_cols.index(base_ch_guess) if base_ch_guess in base_cols else 0))
with c4:
    base_dis_col = st.selectbox("Basistabelle: Baseline-Entladeenergie je 15 min [kWh]", base_cols, index=(base_cols.index(base_dis_guess) if base_dis_guess in base_cols else 0))

c5, c6 = st.columns(2)
with c5:
    price_time_col = st.selectbox("Preisdatei: Zeitstempel", price_cols, index=(price_cols.index(price_time_guess) if price_time_guess in price_cols else 0))
with c6:
    price_val_col = st.selectbox("Preisdatei: Preis", price_cols, index=(price_cols.index(price_val_guess) if price_val_guess in price_cols else 0))

price_unit = st.radio("Preiseinheit", ["€/MWh", "€/kWh", "ct/kWh"], horizontal=True)

# ---------------------- Parameter ----------------------
st.subheader("Batterie-Parameter & Optionen")

colp1, colp2, colp3 = st.columns(3)
with colp1:
    E_max = st.number_input("E_max (nutzbare Kapazität) [kWh]", min_value=1.0, value=150.0, step=1.0)
with colp2:
    P_max = st.number_input("P_max (symmetrisch) [kW]", min_value=0.1, value=100.0, step=1.0)
with colp3:
    rte_pct = st.number_input("Round-trip Efficiency RTE [%]", min_value=1.0, max_value=100.0, value=90.0, step=0.5)

eta_rt = rte_pct / 100.0
eta_ch = math.sqrt(eta_rt)
eta_dis = math.sqrt(eta_rt)

colp4, colp5, colp6 = st.columns(3)
with colp4:
    soc0_pct = st.slider("Start-SoC [% von E_max]", min_value=0, max_value=100, value=0)
with colp5:
    fix_final = st.checkbox("Finaler SoC = Start-SoC (Energie-Bilanz geschlossen)", value=True)
with colp6:
    fees = st.number_input("Gebühren/Handelskosten [€/MWh] (wird von Kauf und Verkauf abgezogen)", min_value=0.0, value=0.0, step=0.1)

# Zyklenlimit pro Kalenderjahr (Equivalent Full Cycles). 0 = kein Limit
colp7, colp8 = st.columns(2)
with colp7:
    cycles_cap = st.number_input("Max. Vollzyklen/Jahr (EFC)", min_value=0.0, value=0.0, step=0.5, help="0 = keine Begrenzung")
with colp8:
    st.write("")

# ---------------------- Datenaufbereitung ----------------------
# Basistabelle
base_df = base_df_raw[[base_time_col, base_soc_col, base_ch_col, base_dis_col]].copy()
base_df.columns = ["ts", "soc_base_kwh", "ch_base_kwh", "dis_base_kwh"]
base_df["ts"] = pd.to_datetime(base_df["ts"], errors="coerce")
base_df = base_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

# Lade-/Entladeleistung aus kWh/15min → kW
# (Bei anderer Intervalllänge rechnen wir dynamisch um)

# Preisdatei
price_df = price_df_raw[[price_time_col, price_val_col]].copy()
price_df.columns = ["ts", "price_raw"]
price_df["ts"] = pd.to_datetime(price_df["ts"], errors="coerce")
price_df = price_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

# Preis in €/MWh normieren
if price_unit == "€/MWh":
    price_df["price_eur_per_mwh"] = price_df["price_raw"].astype(float)
elif price_unit == "€/kWh":
    price_df["price_eur_per_mwh"] = price_df["price_raw"].astype(float) * 1000.0
elif price_unit == "ct/kWh":
    price_df["price_eur_per_mwh"] = price_df["price_raw"].astype(float) * 10.0

# Gebühren von Kauf/Verkauf berücksichtigen: wir ziehen sie in der Zielfunktion je MWh ab
fee_buy = fees  # €/MWh
fee_sell = fees # €/MWh

# Merge auf gemeinsame Zeitachse
DF = pd.merge_asof(
    base_df.sort_values("ts"),
    price_df.sort_values("ts"),
    on="ts",
    direction="nearest",
    tolerance=pd.Timedelta("8min")  # robust gegen 00/15/30/45-Minuten-Raster
)

if DF["price_eur_per_mwh"].isna().any():
    st.error("Preiswerte konnten nicht robust gematcht werden. Bitte prüfen Sie das Zeitraster/Zeitzonen und laden Sie passende 15-min-Preise.")
    st.stop()

# Intervalllänge in Stunden bestimmen (median)
dt = DF["ts"].diff().median()
if pd.isna(dt) or dt == pd.Timedelta(0):
    st.error("Konnte die Zeitschrittweite nicht bestimmen. Bitte Daten prüfen.")
    st.stop()

dt_h = dt.total_seconds() / 3600.0

# Wenn 15-minütig, sind base_kWh je Intervall → kW = kWh / dt_h
DF["p_ch_base_kw"]  = DF["ch_base_kwh"].astype(float)  / dt_h
DF["p_dis_base_kw"] = DF["dis_base_kwh"].astype(float) / dt_h

# Start-SoC absolute kWh
soc0_kwh = E_max * (soc0_pct / 100.0)

# ---------------------- Optimierer ----------------------
def optimize_with_baseline(df: pd.DataFrame,
                           E_max: float, P_max: float,
                           eta_ch: float, eta_dis: float,
                           dt_h: float,
                           soc0_extra_kwh: float = 0.0,
                           fix_final: bool = True,
                           fee_buy: float = 0.0,
                           fee_sell: float = 0.0,
                           cycles_cap: float = 0.0) -> Optional[pd.DataFrame]:
    """Optimiert zusätzliche Arbitrage *oberhalb* der Baseline (Eigenverbrauch).
    Erwartet Spalten: ts, soc_base_kwh, p_ch_base_kw, p_dis_base_kw, price_eur_per_mwh.
    """
    if pulp is None:
        st.error("PuLP ist nicht installiert. Bitte 'pip install pulp' ausführen und App neu starten.")
        return None

    n = len(df)
    m = pulp.LpProblem("BESS_Arbitrage_with_baseline", pulp.LpMaximize)

    ch = pulp.LpVariable.dicts("ch_extra_kw", range(n), lowBound=0)
    dis = pulp.LpVariable.dicts("dis_extra_kw", range(n), lowBound=0)
    soc = pulp.LpVariable.dicts("soc_extra_kwh", range(n), lowBound=0)

    # Zielfunktion
    revenue_terms = []
    for t in range(n):
        p = float(df.loc[t, "price_eur_per_mwh"])  # €/MWh
        revenue_terms.append(((p - fee_sell) * dis[t] - (p + fee_buy) * ch[t]) * (dt_h / 1000.0))
    m += pulp.lpSum(revenue_terms)

    # SoC-Dynamik (nur extra-SoC über Baseline)
    m += soc[0] == soc0_extra_kwh + (eta_ch * ch[0] - (dis[0] / eta_dis)) * dt_h
    for t in range(1, n):
        m += soc[t] == soc[t-1] + (eta_ch * ch[t] - (dis[t] / eta_dis)) * dt_h

    # Headroom: soc_extra <= E_max - soc_base
    for t in range(n):
        headroom = max(0.0, E_max - float(df.loc[t, "soc_base_kwh"]))
        m += soc[t] <= headroom

    # Leistung: baseline + extra <= P_max
    for t in range(n):
        m += ch[t] <= max(0.0, P_max - float(df.loc[t, "p_ch_base_kw"]))
        m += dis[t] <= max(0.0, P_max - float(df.loc[t, "p_dis_base_kw"]))

    # Jahres-Zyklenlimit (EFC)
    if cycles_cap and cycles_cap > 0 and E_max > 0:
        years = df["ts"].dt.year.to_list()
        from collections import defaultdict
        idx_by_year = defaultdict(list)
        for i, y in enumerate(years):
            idx_by_year[y].append(i)
        for y, idxs in idx_by_year.items():
            base_e_dis_y = float(df.iloc[idxs]["p_dis_base_kw"].sum()) * dt_h
            m += pulp.lpSum([dis[i] for i in idxs]) * dt_h + base_e_dis_y <= cycles_cap * E_max

    if fix_final:
        m += soc[n-1] == soc0_extra_kwh

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        st.error(f"Optimierung A (mit Baseline) nicht optimal: {pulp.LpStatus[m.status]}")
        return None

    out = df[["ts", "price_eur_per_mwh", "soc_base_kwh", "p_ch_base_kw", "p_dis_base_kw"]].copy()
    out["p_ch_extra_kw"]  = [pulp.value(ch[t]) for t in range(n)]
    out["p_dis_extra_kw"] = [pulp.value(dis[t]) for t in range(n)]
    out["soc_extra_kwh"]  = [pulp.value(soc[t]) for t in range(n)]

    out["soc_total_kwh"]  = out["soc_base_kwh"] + out["soc_extra_kwh"]
    out["p_ch_total_kw"]  = out["p_ch_base_kw"] + out["p_ch_extra_kw"]
    out["p_dis_total_kw"] = out["p_dis_base_kw"] + out["p_dis_extra_kw"]

    out["e_ch_extra_kwh"]  = out["p_ch_extra_kw"] * dt_h
    out["e_dis_extra_kwh"] = out["p_dis_extra_kw"] * dt_h

    out["revenue_eur"] = ((out["price_eur_per_mwh"] - fee_sell) * out["e_dis_extra_kwh"]
                           - (out["price_eur_per_mwh"] + fee_buy) * out["e_ch_extra_kwh"]) / 1000.0

    return out


def optimize_free(df_price: pd.DataFrame,
                  E_max: float, P_max: float,
                  eta_ch: float, eta_dis: float,
                  dt_h: float,
                  soc0_kwh: float = 0.0,
                  fix_final: bool = True,
                  fee_buy: float = 0.0,
                  fee_sell: float = 0.0,
                  cycles_cap: float = 0.0) -> Optional[pd.DataFrame]:
    """Klassische Arbitrage ohne Baseline-Zwänge. Erwartet ts, price_eur_per_mwh."""
    if pulp is None:
        st.error("PuLP ist nicht installiert. Bitte 'pip install pulp' ausführen und App neu starten.")
        return None

    n = len(df_price)
    m = pulp.LpProblem("BESS_Arbitrage_free", pulp.LpMaximize)

    ch = pulp.LpVariable.dicts("ch_kw", range(n), lowBound=0)
    dis = pulp.LpVariable.dicts("dis_kw", range(n), lowBound=0)
    soc = pulp.LpVariable.dicts("soc_kwh", range(n), lowBound=0, upBound=E_max)

    revenue_terms = []
    for t in range(n):
        p = float(df_price.loc[t, "price_eur_per_mwh"])  # €/MWh
        revenue_terms.append(((p - fee_sell) * dis[t] - (p + fee_buy) * ch[t]) * (dt_h / 1000.0))
    m += pulp.lpSum(revenue_terms)

    m += soc[0] == soc0_kwh + (eta_ch * ch[0] - (dis[0] / eta_dis)) * dt_h
    for t in range(1, n):
        m += soc[t] == soc[t-1] + (eta_ch * ch[t] - (dis[t] / eta_dis)) * dt_h

    for t in range(n):
        m += ch[t] <= P_max
        m += dis[t] <= P_max

    if cycles_cap and cycles_cap > 0 and E_max > 0:
        years = df_price["ts"].dt.year.to_list()
        from collections import defaultdict
        idx_by_year = defaultdict(list)
        for i, y in enumerate(years):
            idx_by_year[y].append(i)
        for y, idxs in idx_by_year.items():
            m += pulp.lpSum([dis[i] for i in idxs]) * dt_h <= cycles_cap * E_max

    if fix_final:
        m += soc[n-1] == soc0_kwh

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        st.error(f"Optimierung B (frei) nicht optimal: {pulp.LpStatus[m.status]}")
        return None

    out = df_price[["ts", "price_eur_per_mwh"]].copy()
    out["p_ch_kw"]  = [pulp.value(ch[t]) for t in range(n)]
    out["p_dis_kw"] = [pulp.value(dis[t]) for t in range(n)]
    out["soc_kwh"]  = [pulp.value(soc[t]) for t in range(n)]

    out["e_ch_kwh"]  = out["p_ch_kw"] * dt_h
    out["e_dis_kwh"] = out["p_dis_kw"] * dt_h

    out["revenue_eur"] = ((out["price_eur_per_mwh"] - fee_sell) * out["e_dis_kwh"]
                           - (out["price_eur_per_mwh"] + fee_buy) * out["e_ch_kwh"]) / 1000.0

    return out

# ---------------------- Optimierung ausführen ----------------------
run = st.button("Optimierung starten", type="primary")

if run:
    # Vorab-Check: Baseline-Zyklen pro Jahr vs. Limit
    if cycles_cap and cycles_cap > 0 and E_max > 0:
        base_cycles_by_year = (DF.assign(year=DF["ts"].dt.year)
                                 .groupby("year")["p_dis_base_kw"].sum() * dt_h / E_max)
        exceeded = base_cycles_by_year[base_cycles_by_year > cycles_cap]
        if not exceeded.empty:
            st.warning("Zyklenlimit bereits durch Baseline überschritten in Jahren: " + ", ".join(str(int(y)) for y in exceeded.index) +
                       ". Die Optimierung wird dann ggf. keine Zusatz-Arbitrage zulassen.")

    # Gemeinsamer Preis-Index (nur Zeit/Preis)
    price_only = DF[["ts", "price_eur_per_mwh"]].copy()

    # Szenario A: mit Baseline
    res_A = optimize_with_baseline(
        DF[["ts", "soc_base_kwh", "p_ch_base_kw", "p_dis_base_kw", "price_eur_per_mwh"]].copy(),
        E_max=E_max, P_max=P_max,
        eta_ch=eta_ch, eta_dis=eta_dis,
        dt_h=dt_h,
        soc0_extra_kwh=0.0,
        fix_final=fix_final,
        fee_buy=fee_buy, fee_sell=fee_sell,
        cycles_cap=cycles_cap
    )

    # Szenario B: frei
    res_B = optimize_free(
        price_only,
        E_max=E_max, P_max=P_max,
        eta_ch=eta_ch, eta_dis=eta_dis,
        dt_h=dt_h,
        soc0_kwh=soc0_kwh,
        fix_final=fix_final,
        fee_buy=fee_buy, fee_sell=fee_sell,
        cycles_cap=cycles_cap
    )

    if (res_A is None) or (res_B is None):
        st.stop()

    # ---------------- KPIs ----------------
    def kpis(df, e_dis_col: str, e_ch_col: str, soc_col: Optional[str] = None):
        rev = df["revenue_eur"].sum()
        e_dis = df[e_dis_col].sum()
        e_ch  = df[e_ch_col].sum()
        efc   = e_dis / E_max if E_max > 0 else np.nan  # Vollzyklen ~ entladene Energie / E_max
        return rev, e_dis, e_ch, efc

    kpiA = kpis(res_A, e_dis_col="e_dis_extra_kwh", e_ch_col="e_ch_extra_kwh")
    kpiB = kpis(res_B, e_dis_col="e_dis_kwh", e_ch_col="e_ch_kwh")

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Erlös A – mit Baseline [€]", f"{kpiA[0]:,.0f}")
        st.caption("Nur zusätzliche Arbitrage oberhalb der Eigenverbrauchs-Baseline.")
    with cB:
        st.metric("Erlös B – frei [€]", f"{kpiB[0]:,.0f}")
        st.caption("Reine Arbitrage ohne weitere Zwänge.")
    with cC:
        st.metric("Delta B − A [€]", f"{(kpiB[0]-kpiA[0]):,.0f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("A: Entladen [MWh]", f"{kpiA[1]/1000:,.2f}")
    with c2:
        st.metric("B: Entladen [MWh]", f"{kpiB[1]/1000:,.2f}")
    with c3:
        st.metric("A: Zusatz-Vollzyklen [#]",  f"{kpiA[3]:,.1f}")

    # ---------------- Plots ----------------
    st.subheader("Zeitverläufe (Ausschnitt)")
    sample = 7*24*int(round(1/dt_h))  # eine Woche
    view = slice(0, min(sample, len(res_A)))

    import altair as alt

    # Preis
    chart_price = alt.Chart(DF.iloc[view]).mark_line().encode(
        x="ts:T", y=alt.Y("price_eur_per_mwh:Q", title="Preis [€/MWh]")
    ).properties(height=200)

    # Leistungen A (extra)
    A_long = res_A.iloc[view][["ts", "p_ch_extra_kw", "p_dis_extra_kw"]].melt("ts", var_name="Signal", value_name="kW")
    chart_A = alt.Chart(A_long).mark_bar().encode(
        x="ts:T", y=alt.Y("kW:Q", title="Leistung A [kW]"), color="Signal:N"
    ).properties(height=200)

    # Leistungen B
    B_long = res_B.iloc[view][["ts", "p_ch_kw", "p_dis_kw"]].melt("ts", var_name="Signal", value_name="kW")
    chart_B = alt.Chart(B_long).mark_bar().encode(
        x="ts:T", y=alt.Y("kW:Q", title="Leistung B [kW]"), color="Signal:N"
    ).properties(height=200)

    st.altair_chart(chart_price, use_container_width=True)
    st.altair_chart(chart_A, use_container_width=True)
    st.altair_chart(chart_B, use_container_width=True)

    # ---------------- Export ----------------
    st.subheader("Export")

    # Zusammenführen und in Excel schreiben
    out = DF[["ts", "price_eur_per_mwh", "soc_base_kwh", "p_ch_base_kw", "p_dis_base_kw"]].copy()
    out = out.merge(res_A[["ts", "p_ch_extra_kw", "p_dis_extra_kw", "soc_extra_kwh", "soc_total_kwh", "e_ch_extra_kwh", "e_dis_extra_kwh", "revenue_eur"]], on="ts", how="left")

    # Szenario B zur gleichen Zeitachse (nur Preis/ts identisch), ggf. Umbenennen der Spalten
    res_B_ren = res_B.rename(columns={
        "p_ch_kw": "p_ch_free_kw",
        "p_dis_kw": "p_dis_free_kw",
        "soc_kwh": "soc_free_kwh",
        "e_ch_kwh": "e_ch_free_kwh",
        "e_dis_kwh": "e_dis_free_kwh",
        "revenue_eur": "revenue_free_eur"
    })
    out = out.merge(res_B_ren[["ts", "p_ch_free_kw", "p_dis_free_kw", "soc_free_kwh", "e_ch_free_kwh", "e_dis_free_kwh", "revenue_free_eur"]], on="ts", how="left")

    # KPI-Sheet
    kpi_df = pd.DataFrame({
        "Kennzahl": [
            "Erlös A mit Baseline [€]",
            "Erlös B frei [€]",
            "Delta B − A [€]",
            "A: Entladen [MWh]",
            "B: Entladen [MWh]",
            "A: Zusatz-Vollzyklen [#]",
            "Zyklenlimit / Jahr [#]"
        ],
        "Wert": [
            round(kpiA[0], 2),
            round(kpiB[0], 2),
            round(kpiB[0]-kpiA[0], 2),
            round(kpiA[1]/1000.0, 3),
            round(kpiB[1]/1000.0, 3),
            round(kpiA[3], 2),
            cycles_cap
        ]
    })

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Zeitreihen")
        kpi_df.to_excel(writer, index=False, sheet_name="KPIs")
    bio.seek(0)

    st.download_button(
        "Ergebnisse als Excel herunterladen",
        data=bio,
        file_name="BESS_Arbitrage_Eigenverbrauch_vs_Frei.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.caption("Wählen Sie die Spalten korrekt zu und klicken Sie auf **Optimierung starten**.")
