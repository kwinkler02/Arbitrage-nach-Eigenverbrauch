# app.py — BESS-Arbitrage (einfach): SoC + belegte Netzlast + belegte BESS-Leistung
#
# ❖ Basistabelle (Excel/CSV): exakt DEINE drei Spalten (+ optional Zeit)
#   1) SoC_kWh             … Baseline-SoC je Slot (unsere Arbitrage darf NICHT darunter fallen)
#   2) Netload_base_kW     … bereits belegte Netzleistung am Anschluss; Vorzeichenkonvention:
#                             + positiv = Einspeisung (Export ins Netz), − negativ = Bezug (Import)
#   3) BESS_busy_kW        … bereits verplante BESS-Leistung (Vorzeichen wie oben: + Entladen, − Laden)
#   (Optional) Zeit        … Zeitstempel (15‑min). Wenn vorhanden, wird mit den Preisen gematcht.
#
# ❖ Preisdatei (Excel/CSV): Zeit, Preis (€/MWh | €/kWh | ct/kWh)
#
# ❖ Idee / Constraints
#   • Entscheidungsvariablen je Slot: ch ≥ 0 (Laden), dis ≥ 0 (Entladen), soc_extra ≥ 0
#   • Grid-Constraint absolut (symmetrischer Netzanschluss P_conn):
#       −P_conn ≤  Netload_base + BESS_busy  + (dis − ch)  ≤ P_conn
#   • Sperr-Slots: wenn |BESS_busy| > ε  ⇒  ch = dis = 0
#   • BESS-Leistung: 0 ≤ ch,dis ≤ P_max (zusätzlich zum Sperr-Flag)
#   • SoC-Dynamik:   soc_extra[t] = soc_extra[t−1] + (η_ch·ch − dis/η_dis)·Δt
#   • SoC-Grenze:    0 ≤ soc_extra ≤ (E_max − SoC_baseline)
#   • Optional:      Jahres‑Zyklenlimit (nur Zusatz-Entladung):  Σ_year(dis·Δt)/E_max ≤ Cap
#   • Zielfunktion:  ∑ ((Preis−Fee_sell)·E_dis − (Preis+Fee_buy)·E_ch)/1000
#
#   → Szenario A: mit Sperren (Slots mit BESS_busy sind gesperrt)
#     Szenario B: frei (Sperren ignoriert, alles andere identisch)

from io import BytesIO
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

try:
    import pulp
except Exception:
    pulp = None

st.set_page_config(page_title="BESS – Arbitrage (SoC + belegte Netzlast)", layout="wide")
st.title("⚡ BESS-Arbitrage mit SoC, belegter Netzlast & gesperrten Slots")

# ---------------------- Uploads ----------------------
cu1, cu2 = st.columns(2)
with cu1:
    base_file = st.file_uploader("Basistabelle (SoC_kWh, Netload_base_kW, BESS_busy_kW, optional Zeit)", type=["xlsx","xls","csv"], key="base")
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

# make a robust guess helper
def guess(colnames, keys):
    low = {c.lower(): c for c in colnames}
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return colnames[0]

has_time_in_base = any(k in " ".join([c.lower() for c in base_cols]) for k in ["zeit","time","date","timestamp"]) 

b_time = None
if has_time_in_base:
    b_time = st.selectbox("Basistabelle: Zeitstempel (optional)", base_cols, index=base_cols.index(guess(base_cols,["zeit","time","date"])) )

b_soc   = st.selectbox("Basistabelle: SoC_kWh (Baseline)", base_cols, index=base_cols.index(guess(base_cols,["soc"])) )
b_nload = st.selectbox("Basistabelle: belegte Netzleistung [kW] (±)", base_cols, index=base_cols.index(guess(base_cols,["net","last","grid","anschluss"])) )
b_busyP = st.selectbox("Basistabelle: BESS_busy_kW (±)", base_cols, index=base_cols.index(guess(base_cols,["bess","busy","lade","entlade","power"])) )

p_time = st.selectbox("Preise: Zeitstempel", price_cols, index=price_cols.index(guess(price_cols,["zeit","time","date"])) )
p_val  = st.selectbox("Preise: Preis-Spalte", price_cols, index=price_cols.index(guess(price_cols,["preis","price","eur","ct"])) )

price_unit = st.radio("Preiseinheit", ["€/MWh","€/kWh","ct/kWh"], horizontal=True)

# ---------------------- Batterie- & Netz-Parameter ----------------------
st.subheader("Batterie- & Netz-Parameter")
cp1, cp2, cp3 = st.columns(3)
with cp1:
    E_max = st.number_input("E_max nutzbar [kWh]", min_value=1.0, value=150.0, step=1.0)
with cp2:
    P_max = st.number_input("BESS P_max (sym) [kW]", min_value=0.1, value=100.0, step=1.0)
with cp3:
    P_conn = st.number_input("Netzanschluss P_conn [kW] (symmetrisch)", min_value=0.1, value=73000.0, step=100.0, help="Max. absolute Netzleistung |P| ≤ P_conn")

cp4, cp5, cp6 = st.columns(3)
with cp4:
    rte_pct = st.number_input("RTE [%]", min_value=1.0, max_value=100.0, value=95.0, step=0.5)
with cp5:
    soc0_extra_pct = st.slider("Start-SoC_extra [% von E_max]", min_value=0, max_value=100, value=0)
with cp6:
    fix_final = st.checkbox("Finaler SoC_extra = Start-SoC_extra", value=True)

cp7, cp8, cp9 = st.columns(3)
with cp7:
    cycles_cap = st.number_input("Max. Vollzyklen/Jahr (nur Zusatz) [#]", min_value=0.0, value=0.0, step=0.5, help="0 = keine Begrenzung")
with cp8:
    fees = st.number_input("Gebühren [€/MWh] (Kauf/Verkauf)", min_value=0.0, value=0.0, step=0.1)
with cp9:
    busy_eps = st.number_input("Sperr-Schwelle |BESS_busy_kW| > ε", min_value=0.0, value=0.0, step=0.1)

eta_rt  = rte_pct/100.0
eta_ch  = eta_rt ** 0.5
eta_dis = eta_rt ** 0.5
soc0_extra_kwh = E_max * (soc0_extra_pct/100.0)
fee_buy = float(fees)
fee_sell = float(fees)

# ---------------------- Datenaufbereitung ----------------------
# Basistabelle
cols = [b_soc, b_nload, b_busyP]
if b_time:
    cols = [b_time] + cols
base = base_raw[cols].copy()
base.columns = (["ts"] if b_time else []) + ["soc_base_kwh","netload_base_kw","bess_busy_kw"]

if b_time:
    base["ts"] = pd.to_datetime(base["ts"], errors="coerce")
    base = base.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

base["soc_base_kwh"]  = pd.to_numeric(base["soc_base_kwh"], errors="coerce").fillna(0.0)
base["netload_base_kw"] = pd.to_numeric(base["netload_base_kw"], errors="coerce").fillna(0.0)
base["bess_busy_kw"]  = pd.to_numeric(base["bess_busy_kw"], errors="coerce").fillna(0.0)

# Preise
price = price_raw[[p_time, p_val]].copy()
price.columns = ["ts","price_raw"]
price["ts"] = pd.to_datetime(price["ts"], errors="coerce")
price = price.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

if price_unit == "€/MWh":
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce")
elif price_unit == "€/kWh":
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 1000.0
else:  # ct/kWh
    price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 10.0

# Join Preise
if b_time:
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
else:
    # Kein Zeitstempel: nach Reihenfolge matchen (Längen müssen passen)
    if len(base) != len(price):
        st.error("Ohne Zeitstempel müssen Basistabelle und Preise gleich lang sein.")
        st.stop()
    DF = base.copy()
    DF["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce").values
    dt_h = st.number_input("Zeitschrittlänge [h]", min_value=0.01, value=0.25, step=0.01)

# Busy-Flag aus BESS_busy_kW ableiten
DF["busy"] = (DF["bess_busy_kw"].abs() > busy_eps).astype(int)

# Vorwarnung: Basislast verletzt ggf. Netzanschluss
over = (DF["netload_base_kw"] + DF["bess_busy_kw"]).abs() > P_conn + 1e-9
if over.any():
    st.warning("Warnung: In manchen Slots überschreitet (Netload_base + BESS_busy) bereits den Netzanschluss. Die Optimierung kann hier keine zusätzliche Leistung mehr zuweisen.")

# Baseline-Energie (aus belegter BESS-Leistung):
DF["e_dis_base_kwh"] = np.maximum(DF["bess_busy_kw"], 0.0) * dt_h
DF["e_ch_base_kwh"]  = np.maximum(-DF["bess_busy_kw"], 0.0) * dt_h

# ---------------------- Optimierer ----------------------
from collections import defaultdict

def optimize(df: pd.DataFrame,
             E_max: float, P_max: float, P_conn: float,
             eta_ch: float, eta_dis: float,
             dt_h: float,
             soc0_extra_kwh: float,
             fix_final: bool,
             fee_buy: float, fee_sell: float,
             cycles_cap: float,
             ignore_busy: bool = False) -> Optional[pd.DataFrame]:
    if pulp is None:
        st.error("PuLP fehlt: pip install pulp (und ggf. coinor-cbc)")
        return None

    n = len(df)
    m = pulp.LpProblem("BESS_SOC_CONN", pulp.LpMaximize)

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

    # SoC-Grenze
    for i in range(n):
        headroom_soc = max(0.0, E_max - float(df.loc[i, "soc_base_kwh"]))
        m += soc[i] <= headroom_soc

    # Leistung + Sperre + Netzanschluss (absolut)
    for i in range(n):
        busy = int(df.loc[i, "busy"]) if not ignore_busy else 0
        cap_bess = P_max * (1 - busy)
        m += ch[i]  <= cap_bess
        m += dis[i] <= cap_bess

        base_net = float(df.loc[i, "netload_base_kw"]) + float(df.loc[i, "bess_busy_kw"])  # kW
        # −P_conn ≤ base_net + (dis − ch) ≤ P_conn
        m += base_net + (dis[i] - ch[i]) <= P_conn
        m += -base_net - (dis[i] - ch[i]) <= P_conn

    # Zyklenlimit (pro Kalenderjahr, falls Zeit vorhanden)
    if cycles_cap and cycles_cap > 0 and ("ts" in df.columns) and (df["ts"].notna().all()) and E_max > 0:
        idx_by_year = defaultdict(list)
        years = df["ts"].dt.year.to_list()
        for i, y in enumerate(years):
            idx_by_year[y].append(i)
        for y, idxs in idx_by_year.items():
            m += pulp.lpSum(dis[i] for i in idxs) * dt_h <= cycles_cap * E_max
    elif cycles_cap and cycles_cap > 0 and E_max > 0:
        # Fallback ohne Zeit: über gesamten Horizont begrenzen
        m += pulp.lpSum(dis[i] for i in range(n)) * dt_h <= cycles_cap * E_max

    if fix_final:
        m += soc[n-1] == soc0_extra_kwh

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        st.error(f"Optimierung nicht optimal: {pulp.LpStatus[m.status]}")
        return None

    out = df[[c for c in ["ts","soc_base_kwh","netload_base_kw","bess_busy_kw","busy","price_eur_per_mwh"] if c in df.columns]].copy()
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
res_A = optimize(DF.copy(), E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
                 soc0_extra_kwh, fix_final, fee_buy, fee_sell, cycles_cap, ignore_busy=False)
# B: frei (Sperren ignorieren)
res_B = optimize(DF.copy(), E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
                 soc0_extra_kwh, fix_final, fee_buy, fee_sell, cycles_cap, ignore_busy=True)

if res_A is None or res_B is None:
    st.stop()

# ---------------------- KPIs ----------------------
revA = res_A["revenue_eur"].sum()
revB = res_B["revenue_eur"].sum()

# Gesamt‑Vollzyklen: Baseline‑Entladung + Zusatz (A), bzw. nur Optimierung (B)
cycles_total_A = (DF["e_dis_base_kwh"].sum() + res_A["e_dis_kwh"].sum())/E_max if E_max>0 else np.nan
cycles_total_B = (res_B["e_dis_kwh"].sum())/E_max if E_max>0 else np.nan

c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Erlös A – mit Sperren [€]", f"{revA:,.0f}")
with c2:
    st.metric("Erlös B – frei [€]", f"{revB:,.0f}")
with c3:
    st.metric("Delta B − A [€]", f"{(revB-revA):,.0f}")

c4,c5 = st.columns(2)
with c4:
    st.metric("A: Gesamt‑Vollzyklen [#]", f"{cycles_total_A:,.1f}")
with c5:
    st.metric("B: Gesamt‑Vollzyklen [#]", f"{cycles_total_B:,.1f}")

# ---------------------- Plots (Ausschnitt) ----------------------
import altair as alt

sample = int(7*24/max(dt_h, 1e-6))
slA = res_A.iloc[:min(sample, len(res_A))]

if "ts" in slA.columns:
    xenc = "ts:T"
else:
    slA = slA.reset_index().rename(columns={"index":"slot"})
    xenc = "slot:Q"

chart_price = alt.Chart(slA).mark_line().encode(
    x=xenc, y=alt.Y("price_eur_per_mwh:Q", title="Preis [€/MWh]")
).properties(height=200)

A_long = slA[[c for c in slA.columns if c in ["ts","slot"]] + ["p_ch_kw","p_dis_kw"]].melt(
    id_vars=[c for c in slA.columns if c in ["ts","slot"]], var_name="Signal", value_name="kW")
chart_A = alt.Chart(A_long).mark_bar().encode(
    x=xenc, y=alt.Y("kW:Q", title="Leistung A [kW]"), color="Signal:N"
).properties(height=200)

soc_view = slA[[c for c in slA.columns if c in ["ts","slot"]] + ["soc_total_kwh"]]
chart_soc = alt.Chart(soc_view).mark_line().encode(
    x=xenc, y=alt.Y("soc_total_kwh:Q", title="SoC gesamt [kWh]")
).properties(height=200)

st.altair_chart(chart_price, use_container_width=True)
st.altair_chart(chart_A, use_container_width=True)
st.altair_chart(chart_soc, use_container_width=True)

# ---------------------- Export ----------------------
# Zeitachse möglichst mitgeben
join_cols = [c for c in ["ts"] if c in DF.columns]
base_out = DF[join_cols + ["soc_base_kwh","netload_base_kw","bess_busy_kw","busy","price_eur_per_mwh","e_dis_base_kwh","e_ch_base_kwh"]].copy()

out = base_out.merge(
    res_A[[c for c in ["ts"] if c in res_A.columns] + ["p_ch_kw","p_dis_kw","soc_extra_kwh","soc_total_kwh","e_ch_kwh","e_dis_kwh","revenue_eur"]]
      .rename(columns={"p_ch_kw":"p_ch_A_kw","p_dis_kw":"p_dis_A_kw","soc_extra_kwh":"soc_extra_A_kwh","soc_total_kwh":"soc_total_A_kwh",
                       "e_ch_kwh":"e_ch_A_kwh","e_dis_kwh":"e_dis_A_kwh","revenue_eur":"revenue_A_eur"}),
    on="ts" if "ts" in res_A.columns else None, how="left")

if "ts" in res_B.columns:
    out = out.merge(
        res_B[["ts","p_ch_kw","p_dis_kw","soc_extra_kwh","soc_total_kwh","e_ch_kwh","e_dis_kwh","revenue_eur"]]
          .rename(columns={"p_ch_kw":"p_ch_B_kw","p_dis_kw":"p_dis_B_kw","soc_extra_kwh":"soc_extra_B_kwh","soc_total_kwh":"soc_total_B_kwh",
                           "e_ch_kwh":"e_ch_B_kwh","e_dis_kwh":"e_dis_B_kwh","revenue_eur":"revenue_B_eur"}),
        on="ts", how="left")
else:
    out = out.join(
        res_B[["p_ch_kw","p_dis_kw","soc_extra_kwh","soc_total_kwh","e_ch_kwh","e_dis_kwh","revenue_eur"]]
          .rename(columns={"p_ch_kw":"p_ch_B_kw","p_dis_kw":"p_dis_B_kw","soc_extra_kwh":"soc_extra_B_kwh","soc_total_kwh":"soc_total_B_kwh",
                           "e_ch_kwh":"e_ch_B_kwh","e_dis_kwh":"e_dis_B_kwh","revenue_eur":"revenue_B_eur"})
    )

kpi_df = pd.DataFrame({
    "Kennzahl": [
        "Erlös A [€]","Erlös B [€]","Delta B−A [€]",
        "A: Gesamt‑Vollzyklen [#]","B: Gesamt‑Vollzyklen [#]",
        "RTE [%]","P_max [kW]","E_max [kWh]","P_conn [kW]","Busy‑ε [kW]"
    ],
    "Wert": [round(revA,2), round(revB,2), round(revB-revA,2), round(cycles_total_A,2), round(cycles_total_B,2), rte_pct, P_max, E_max, P_conn, busy_eps]
})

bio = BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as w:
    out.to_excel(w, index=False, sheet_name="Zeitreihen")
    kpi_df.to_excel(w, index=False, sheet_name="KPIs")

bio.seek(0)

st.download_button(
    "Ergebnisse als Excel herunterladen",
    data=bio,
    file_name="BESS_Arbitrage_SoC_Netload.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
