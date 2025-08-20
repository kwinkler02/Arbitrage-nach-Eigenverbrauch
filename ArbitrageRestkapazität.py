# app.py — BESS-Arbitrage (optimiert): SoC + belegte Netzlast + belegte BESS-Leistung
#
# ❖ Basistabelle (Excel/CSV): exakt DEINE drei Spalten (+ optional Zeit)
#   1) SoC_kWh             … Baseline-SoC je Slot (unsere Arbitrage darf NICHT darunter fallen)
#   2) Netload_base_kW     … bereits belegte Netzleistung am Anschluss; Vorzeichenkonvention:
#                             + positiv = Einspeisung (Export ins Netz), − negativ = Bezug (Import)
#   3) BESS_busy_kW        … bereits verplante BESS-Leistung (Vorzeichen wie oben: + Entladen, − Laden)
#   (Optional) Zeit        … Zeitstempel (15‑min). Wenn vorhanden, wird mit den Preisen gematcht.
#
# ❖ Preisdatei (Excel/CSV): Zeit, Preis (€/MWh | €/kWh | ct/kWh)

from io import BytesIO
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

try:
    import pulp
except Exception:
    pulp = None

st.set_page_config(page_title="BESS – Arbitrage (optimiert)", layout="wide")
st.title("⚡ BESS-Arbitrage mit SoC, belegter Netzlast & intelligenter Restkapazitätsnutzung")

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
        return pd.read_csv(f, encoding='utf-8')
    return pd.read_excel(f, sheet_name=0)

try:
    base_raw = read_any(base_file)
    price_raw = read_any(price_file)
except Exception as e:
    st.error(f"Fehler beim Einlesen der Dateien: {str(e)}")
    st.stop()

# ---------------------- Column mapping ----------------------
st.subheader("Spaltenzuordnung")
base_cols  = list(base_raw.columns)
price_cols = list(price_raw.columns)

# Robuste Spaltenerkennung
def guess(colnames, keys):
    low = {c.lower(): c for c in colnames}
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return colnames[0] if colnames else None

# Zeitstempel-Erkennung in Basistabelle
has_time_in_base = any(k in " ".join([c.lower() for c in base_cols]) for k in ["zeit","time","date","timestamp"]) 

b_time = None
if has_time_in_base:
    b_time = st.selectbox("Basistabelle: Zeitstempel (optional)", [None] + base_cols, 
                         index=base_cols.index(guess(base_cols,["zeit","time","date"])) + 1 if guess(base_cols,["zeit","time","date"]) else 0)

b_soc   = st.selectbox("Basistabelle: SoC_kWh (Baseline)", base_cols, 
                      index=base_cols.index(guess(base_cols,["soc"])) if guess(base_cols,["soc"]) else 0)
b_nload = st.selectbox("Basistabelle: belegte Netzleistung [kW] (±)", base_cols, 
                      index=base_cols.index(guess(base_cols,["net","last","grid","anschluss"])) if guess(base_cols,["net","last","grid","anschluss"]) else 0)
b_busyP = st.selectbox("Basistabelle: BESS_busy_kW (±)", base_cols, 
                      index=base_cols.index(guess(base_cols,["bess","busy","lade","entlade","power"])) if guess(base_cols,["bess","busy","lade","entlade","power"]) else 0)

p_time = st.selectbox("Preise: Zeitstempel", price_cols, 
                     index=price_cols.index(guess(price_cols,["zeit","time","date"])) if guess(price_cols,["zeit","time","date"]) else 0)
p_val  = st.selectbox("Preise: Preis-Spalte", price_cols, 
                     index=price_cols.index(guess(price_cols,["preis","price","eur","ct"])) if guess(price_cols,["preis","price","eur","ct"]) else 0)

price_unit = st.radio("Preiseinheit", ["€/MWh","€/kWh","ct/kWh"], horizontal=True)

# ---------------------- Batterie- & Netz-Parameter ----------------------
st.subheader("Batterie- & Netz-Parameter")
cp1, cp2, cp3 = st.columns(3)
with cp1:
    E_max = st.number_input("E_max nutzbar [kWh]", min_value=1.0, value=150.0, step=1.0)
with cp2:
    P_max = st.number_input("BESS P_max (sym) [kW]", min_value=0.1, value=100.0, step=1.0)
with cp3:
    P_conn = st.number_input("Netzanschluss P_conn [kW] (symmetrisch)", min_value=0.1, value=73.0, step=1.0, 
                           help="Max. absolute Netzleistung |P| ≤ P_conn")

# Validierung der Parameter
if P_max > P_conn:
    st.warning("⚠️ BESS-Leistung > Netzanschluss - kann zu suboptimalen Lösungen führen")

cp4, cp5, cp6 = st.columns(3)
with cp4:
    rte_pct = st.number_input("RTE [%]", min_value=50.0, max_value=100.0, value=95.0, step=0.5)
with cp5:
    soc0_extra_pct = st.slider("Start-SoC_extra [% von E_max]", min_value=0, max_value=100, value=0)
with cp6:
    fix_final = st.checkbox("Finaler SoC_extra = Start-SoC_extra", value=True)

cp7, cp8, cp9 = st.columns(3)
with cp7:
    cycles_cap = st.number_input(
        "Max. Gesamt-Vollzyklen/Jahr [#]",
        min_value=0.0, value=0.0, step=0.5,
        help="0 = keine Begrenzung; gilt für Baseline + Zusatz"
    )
with cp8:
    fees = st.number_input("Gebühren [€/MWh] (Kauf/Verkauf)", min_value=0.0, value=0.0, step=0.1)
with cp9:
    busy_eps = st.number_input("Sperr-Schwelle |BESS_busy_kW| > ε", min_value=0.0, value=0.1, step=0.1)

# Vereinfachte Optimierungsoptionen
st.subheader("Optimierungsoptionen")
op1, op2 = st.columns(2)
with op1:
    ignore_busy = st.checkbox("Sperr-Slots ignorieren (Was-wäre-wenn-Szenario)", value=False,
                             help="Zeigt das theoretische Maximum ohne Berücksichtigung bereits belegter Slots")
with op2:
    cycles_constraint = st.radio("Zyklenlimit-Behandlung", 
                               ["Obergrenze", "Exakte Ausnutzung", "Ignorieren"], 
                               horizontal=True,
                               help="Obergrenze: ≤ Limit, Exakte Ausnutzung: = Limit")

# Parameter berechnen
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

# Datenvalidierung und -bereinigung
try:
    if b_time:
        base["ts"] = pd.to_datetime(base["ts"], errors="coerce")
        if base["ts"].isna().any():
            st.error("Zeitstempel in Basistabelle konnten nicht konvertiert werden!")
            st.stop()
        base = base.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    base["soc_base_kwh"]  = pd.to_numeric(base["soc_base_kwh"], errors="coerce").fillna(0.0)
    base["netload_base_kw"] = pd.to_numeric(base["netload_base_kw"], errors="coerce").fillna(0.0)
    base["bess_busy_kw"]  = pd.to_numeric(base["bess_busy_kw"], errors="coerce").fillna(0.0)

    # SoC-Plausibilität prüfen
    if (base["soc_base_kwh"] > E_max).any():
        st.error("❌ SoC_baseline übersteigt E_max in manchen Slots!")
        st.stop()
    
    if (base["soc_base_kwh"] < 0).any():
        st.warning("⚠️ Negative SoC_baseline-Werte gefunden - werden auf 0 gesetzt")
        base["soc_base_kwh"] = base["soc_base_kwh"].clip(lower=0)

except Exception as e:
    st.error(f"Fehler bei Basistabellen-Verarbeitung: {str(e)}")
    st.stop()

# Preise
try:
    price = price_raw[[p_time, p_val]].copy()
    price.columns = ["ts","price_raw"]
    price["ts"] = pd.to_datetime(price["ts"], errors="coerce")
    if price["ts"].isna().any():
        st.error("Zeitstempel in Preisdatei konnten nicht konvertiert werden!")
        st.stop()
    price = price.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # Preiskonvertierung
    if price_unit == "€/MWh":
        price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce")
    elif price_unit == "€/kWh":
        price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 1000.0
    else:  # ct/kWh
        price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 10.0

    # Preisplausibilität
    if (price["price_eur_per_mwh"] < -1000).any() or (price["price_eur_per_mwh"] > 1000).any():
        st.warning("⚠️ Ungewöhnliche Preise detected - bitte Einheit prüfen")
    
except Exception as e:
    st.error(f"Fehler bei Preis-Verarbeitung: {str(e)}")
    st.stop()

# Join Preise
if b_time:
    try:
        DF = pd.merge_asof(base.sort_values("ts"), price.sort_values("ts"), 
                          on="ts", direction="nearest", tolerance=pd.Timedelta("8min"))
        if DF["price_eur_per_mwh"].isna().any():
            st.error("❌ Preiswerte konnten nicht gematcht werden. Bitte Zeitraster/Zeitzone prüfen.")
            st.stop()
        
        # Zeitschritt berechnen
        time_diffs = DF["ts"].diff().dropna()
        if time_diffs.empty:
            st.error("Konnte Zeitschrittweite nicht bestimmen - zu wenig Datenpunkte.")
            st.stop()
        
        dt_h = time_diffs.median().total_seconds()/3600.0
        if dt_h <= 0:
            st.error("Ungültige Zeitschrittweite berechnet.")
            st.stop()
            
    except Exception as e:
        st.error(f"Fehler beim Zusammenführen der Daten: {str(e)}")
        st.stop()
else:
    # Ohne Zeitstempel: Längen müssen übereinstimmen
    if len(base) != len(price):
        st.error(f"❌ Ohne Zeitstempel müssen Basistabelle ({len(base)}) und Preise ({len(price)}) gleich lang sein.")
        st.stop()
    DF = base.copy()
    DF["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce").values
    dt_h = st.number_input("Zeitschrittlänge [h]", min_value=0.01, max_value=24.0, value=0.25, step=0.01)

# Performance-Warnung
if len(DF) > 8760:  # > 1 Jahr
    st.warning(f"⚠️ Große Datenmenge ({len(DF)} Zeitschritte) - Optimierung kann mehrere Minuten dauern")

# Busy-Flag und Validierungen
DF["busy"] = (DF["bess_busy_kw"].abs() > busy_eps).astype(int)

# Netzanschluss-Vorwarnung
net_total_current = DF["netload_base_kw"] + DF["bess_busy_kw"]
over_slots = (net_total_current.abs() > P_conn + 1e-6).sum()
if over_slots > 0:
    st.warning(f"⚠️ In {over_slots} Slots überschreitet die Basislast bereits den Netzanschluss. "
               f"Arbitrage hat dort keinen Spielraum.")

# Baseline-Energie berechnen
DF["e_dis_base_kwh"] = np.maximum(DF["bess_busy_kw"], 0.0) * dt_h
DF["e_ch_base_kwh"]  = np.maximum(-DF["bess_busy_kw"], 0.0) * dt_h

# Zyklenlimit-Vorprüfung
if cycles_cap and cycles_cap > 0 and E_max > 0:
    base_cycles_total = DF["e_dis_base_kwh"].sum() / E_max
    if base_cycles_total > cycles_cap:
        st.error(f"❌ Zyklenlimit bereits durch Baseline überschritten: "
                f"{base_cycles_total:.2f} > {cycles_cap}. Keine Arbitrage möglich.")
        if not ignore_busy:
            st.stop()

# ---------------------- Optimierer ----------------------
def optimize_bess(df: pd.DataFrame,
                  E_max: float, P_max: float, P_conn: float,
                  eta_ch: float, eta_dis: float, dt_h: float,
                  soc0_extra_kwh: float, fix_final: bool,
                  fee_buy: float, fee_sell: float,
                  cycles_cap: float, cycles_constraint: str,
                  ignore_busy: bool = False) -> Optional[pd.DataFrame]:
    
    if pulp is None:
        st.error("❌ PuLP nicht verfügbar. Bitte installieren: pip install pulp")
        return None

    n = len(df)
    if n == 0:
        st.error("Keine Daten für Optimierung verfügbar!")
        return None

    # Problem definieren
    prob = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)

    # Variablen
    ch  = pulp.LpVariable.dicts("ch_kw",  range(n), lowBound=0)
    dis = pulp.LpVariable.dicts("dis_kw", range(n), lowBound=0)
    soc = pulp.LpVariable.dicts("soc_extra_kwh", range(n), lowBound=0)

    # Zielfunktion: Erlös aus Arbitrage
    prob += pulp.lpSum(
        ((float(df.loc[i,"price_eur_per_mwh"]) - fee_sell) * dis[i] - 
         (float(df.loc[i,"price_eur_per_mwh"]) + fee_buy) * ch[i]) * (dt_h/1000.0)
        for i in range(n)
    )

    # SoC-Dynamik
    prob += soc[0] == soc0_extra_kwh + (eta_ch*ch[0] - (dis[0]/eta_dis)) * dt_h
    for i in range(1, n):
        prob += soc[i] == soc[i-1] + (eta_ch*ch[i] - (dis[i]/eta_dis)) * dt_h

    # Constraints
    for i in range(n):
        # SoC-Grenzen (verfügbarer Headroom)
        soc_base_i = float(df.loc[i, "soc_base_kwh"])
        headroom = max(0.0, E_max - soc_base_i)
        prob += soc[i] <= headroom

        # BESS-Leistungsgrenzen mit Sperr-Logik
        if not ignore_busy:
            busy_i = int(df.loc[i, "busy"])
            available_power = P_max * (1 - busy_i)
        else:
            available_power = P_max
        
        prob += ch[i]  <= available_power
        prob += dis[i] <= available_power

        # Netzanschluss-Constraints (vereinfacht)
        base_net = float(df.loc[i, "netload_base_kw"]) + float(df.loc[i, "bess_busy_kw"])
        net_total = base_net + (dis[i] - ch[i])
        prob += net_total <= P_conn
        prob += net_total >= -P_conn

    # Zyklenlimit
    if cycles_cap and cycles_cap > 0 and E_max > 0 and cycles_constraint != "Ignorieren":
        base_cycles = float(df["e_dis_base_kwh"].sum()) if not ignore_busy else 0.0
        additional_cycles = pulp.lpSum(dis[i] for i in range(n)) * dt_h
        
        if cycles_constraint == "Exakte Ausnutzung":
            prob += additional_cycles + base_cycles == cycles_cap * E_max
        else:  # Obergrenze
            prob += additional_cycles + base_cycles <= cycles_cap * E_max

    # End-SoC Constraint
    if fix_final:
        prob += soc[n-1] == soc0_extra_kwh

    # Lösen
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[prob.status] != "Optimal":
            st.error(f"❌ Optimierung fehlgeschlagen: {pulp.LpStatus[prob.status]}")
            return None

    except Exception as e:
        st.error(f"❌ Fehler bei Optimierung: {str(e)}")
        return None

    # Ergebnisse extrahieren
    result = df.copy()
    result["p_ch_kw"]  = [pulp.value(ch[i]) or 0.0 for i in range(n)]
    result["p_dis_kw"] = [pulp.value(dis[i]) or 0.0 for i in range(n)]
    result["soc_extra_kwh"] = [pulp.value(soc[i]) or 0.0 for i in range(n)]

    # Berechnete Felder
    result["soc_total_kwh"] = result["soc_base_kwh"] + result["soc_extra_kwh"]
    result["e_ch_kwh"]  = result["p_ch_kw"]  * dt_h
    result["e_dis_kwh"] = result["p_dis_kw"] * dt_h
    result["net_total_kw"] = (result["netload_base_kw"] + result["bess_busy_kw"] + 
                             result["p_dis_kw"] - result["p_ch_kw"])
    result["revenue_eur"] = ((result["price_eur_per_mwh"] - fee_sell) * result["e_dis_kwh"] - 
                            (result["price_eur_per_mwh"] + fee_buy) * result["e_ch_kwh"]) / 1000.0

    return result

# ---------------------- Optimierung ausführen ----------------------
run_button = st.button("🚀 Optimierung starten", type="primary")
if not run_button:
    st.stop()

with st.spinner("Optimierung läuft..."):
    result = optimize_bess(
        DF.copy(), E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
        soc0_extra_kwh, fix_final, fee_buy, fee_sell, 
        cycles_cap, cycles_constraint, ignore_busy
    )

if result is None:
    st.stop()

# ---------------------- Ergebnisse & KPIs ----------------------
st.subheader("📊 Ergebnisse")

revenue_total = result["revenue_eur"].sum()
energy_charged = result["e_ch_kwh"].sum()
energy_discharged = result["e_dis_kwh"].sum()
arbitrage_cycles = energy_discharged / E_max if E_max > 0 else 0
baseline_cycles = DF["e_dis_base_kwh"].sum() / E_max if E_max > 0 else 0
total_cycles = arbitrage_cycles + baseline_cycles

# Metriken anzeigen
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gesamt-Erlös", f"{revenue_total:,.0f} €")
with col2:
    st.metric("Arbitrage-Zyklen", f"{arbitrage_cycles:.1f}")
with col3:
    st.metric("Gesamt-Zyklen", f"{total_cycles:.1f}")
with col4:
    efficiency = (energy_discharged / energy_charged * 100) if energy_charged > 0 else 0
    st.metric("Effizienz", f"{efficiency:.1f}%")

# Zusätzliche KPIs
col5, col6, col7, col8 = st.columns(4)
with col5:
    avg_price_charge = (result[result["e_ch_kwh"] > 0]["price_eur_per_mwh"] * result[result["e_ch_kwh"] > 0]["e_ch_kwh"]).sum() / energy_charged if energy_charged > 0 else 0
    st.metric("⌀ Lade-Preis", f"{avg_price_charge:.1f} €/MWh")
with col6:
    avg_price_discharge = (result[result["e_dis_kwh"] > 0]["price_eur_per_mwh"] * result[result["e_dis_kwh"] > 0]["e_dis_kwh"]).sum() / energy_discharged if energy_discharged > 0 else 0
    st.metric("⌀ Entlade-Preis", f"{avg_price_discharge:.1f} €/MWh")
with col7:
    max_net_load = result["net_total_kw"].abs().max()
    st.metric("Max. Netzlast", f"{max_net_load:.1f} kW")
with col8:
    active_slots = (result["p_ch_kw"] + result["p_dis_kw"] > 0.1).sum()
    st.metric("Aktive Slots", f"{active_slots} / {len(result)}")

# ---------------------- Visualisierung ----------------------
st.subheader("📈 Zeitreihendarstellung (erste 7 Tage)")

# Daten für Plot vorbereiten (erstes Woche)
sample_hours = min(7*24, len(result))
sample_slots = max(1, int(sample_hours / dt_h))
plot_data = result.iloc[:sample_slots].copy()

if "ts" in plot_data.columns:
    x_axis = "ts:T"
    plot_data = plot_data.reset_index()
else:
    plot_data = plot_data.reset_index()
    x_axis = "index:Q"

# Charts mit Altair
import altair as alt

# Preise
price_chart = alt.Chart(plot_data).mark_line(color='blue').encode(
    x=alt.X(x_axis, title="Zeit"),
    y=alt.Y("price_eur_per_mwh:Q", title="Preis [€/MWh]")
).properties(height=150, title="Day-Ahead Preise")

# BESS-Leistung
power_data = plot_data[["index" if "ts" not in plot_data.columns else "ts", "p_ch_kw", "p_dis_kw"]].melt(
    id_vars=["index" if "ts" not in plot_data.columns else "ts"],
    value_vars=["p_ch_kw", "p_dis_kw"],
    var_name="Typ", value_name="Leistung"
)

power_chart = alt.Chart(power_data).mark_bar().encode(
    x=alt.X(x_axis.split(':')[0] + ":O" if "index" in x_axis else x_axis, title="Zeit"),
    y=alt.Y("Leistung:Q", title="BESS-Leistung [kW]"),
    color=alt.Color("Typ:N", scale=alt.Scale(domain=["p_ch_kw", "p_dis_kw"], 
                                           range=["red", "green"]))
).properties(height=200, title="BESS Lade-/Entladeleistung")

# SoC
soc_chart = alt.Chart(plot_data).mark_line(color='orange', strokeWidth=2).encode(
    x=alt.X(x_axis, title="Zeit"),
    y=alt.Y("soc_total_kwh:Q", title="SoC [kWh]", scale=alt.Scale(domain=[0, E_max]))
).properties(height=150, title="Batterie-SoC (gesamt)")

# Netzlast
net_chart = alt.Chart(plot_data).mark_line(color='purple').encode(
    x=alt.X(x_axis, title="Zeit"),
    y=alt.Y("net_total_kw:Q", title="Netzlast [kW]")
).properties(height=150, title="Gesamte Netzlast")

# Charts anzeigen
st.altair_chart(price_chart, use_container_width=True)
st.altair_chart(power_chart, use_container_width=True)
st.altair_chart(soc_chart, use_container_width=True)
st.altair_chart(net_chart, use_container_width=True)

# ---------------------- Export ----------------------
st.subheader("💾 Export")

# KPI-Zusammenfassung
kpi_summary = pd.DataFrame({
    "Kennzahl": [
        "Gesamt-Erlös [€]", "Energie geladen [kWh]", "Energie entladen [kWh]",
        "Arbitrage-Zyklen [#]", "Baseline-Zyklen [#]", "Gesamt-Zyklen [#]",
        "Roundtrip-Effizienz [%]", "Ø Lade-Preis [€/MWh]", "Ø Entlade-Preis [€/MWh]",
        "Max. Netzlast [kW]", "Aktive Zeitslots [#]", "RTE [%]", "E_max [kWh]",
        "P_max [kW]", "P_conn [kW]", "Zyklenlimit [#/Jahr]"
    ],
    "Wert": [
        round(revenue_total, 2), round(energy_charged, 1), round(energy_discharged, 1),
        round(arbitrage_cycles, 2), round(baseline_cycles, 2), round(total_cycles, 2),
        round(efficiency, 1), round(avg_price_charge, 1), round(avg_price_discharge, 1),
        round(max_net_load, 1), active_slots, rte_pct, E_max,
        P_max, P_conn, cycles_cap
    ]
})

# Export vorbereiten
export
