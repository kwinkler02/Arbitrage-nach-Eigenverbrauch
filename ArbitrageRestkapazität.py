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

# Vereinfachte Optimierungsoptionen
st.subheader("Optimierungsoptionen")
ignore_busy = st.checkbox("Baseline ignorieren (Was-wäre-wenn-Szenario)", value=False,
                         help="Zeigt das theoretische Maximum ohne Berücksichtigung bereits belegter Kapazitäten")

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

# Netzanschluss-Vorwarnung
net_total_current = DF["netload_base_kw"] + DF["bess_busy_kw"]
over_slots =
