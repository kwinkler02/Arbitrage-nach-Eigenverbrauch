# Zusätzliche Detailmetriken
st.write("### 🔍 Weitere Analysen")
col8, col9, col10, col11 = st.columns(4)
with col8:
    avg_price_charge = (result[result["e_ch_kwh"] > 0]["price_eur_per_mwh"] * result[result["e_ch_kwh"] > 0]["e_ch_kwh"]).sum() / energy_charged if energy_charged > 0 else 0
    avg_price_charge_free = (result_free[result_free["e_ch_free_kwh"] > 0]["price_eur_per_mwh"] * result_free[result_free["e_ch_free_kwh"] > 0]["e_ch_free_kwh"]).sum() / energy_charged_free if energy_charged_free > 0 else 0
    st.metric("⌀ Lade-Preis (real)", f"{avg_price_charge:.1f} €/MWh")
    st.metric("⌀ Lade-Preis (frei)", f"{avg_price_charge_free:.1f} €/MWh")
with col9:
    avg_price_discharge = (result[result["e_dis_kwh"] > 0]["price_eur_per_mwh"] * result[result["e_dis_kwh"] > 0]["e_dis_kwh"]).sum() / energy_discharged if energy_discharged > 0 else 0
    avg_price_discharge_free = (result_free[result_free["e_dis_free_kwh"] > 0]["price_eur_per_mwh"] * result_free[result_free["e_dis_free_kwh"] > 0]["e_dis_free_kwh"]).sum() / energy_discharged_free if energy_discharged_free > 0 else 0
    st.metric("⌀ Entlade-Preis (real)", f"{avg_price_discharge:.1f} €/MWh")
    st.metric("⌀ Entlade-Preis (frei)", f"{avg_price_discharge_free:.1f} €/MWh")
with col10:
    max_net_load = result["net_total_kw"].abs().max()
    st.metric("Max. Netzlast", f"{max_net_load:.1f} kW")
    st.metric("Netz-Constraint", f"± {P_conn:.0f} kW")
with col11:
    price_spread = avg_price_discharge - avg_price_charge if avg_price_charge > 0 else 0
    price_spread_free = avg_price_discharge_free - avg_price_charge_free if avg_price_charge_free > 0 else 0
    st.metric("Preis-Spread (real)", f"{price_spread:.1f} €/MWh")
    st.metric("Preis-Spread (frei)", f"{price_spread_free:.1f} €/MWh")# app.py — BESS-Arbitrage (optimiert): SoC + belegte Netzlast + belegte BESS-Leistung
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

# Busy-Flag entfernt - wir nutzen verfügbare Restkapazität optimal
# DF["busy"] = (DF["bess_busy_kw"].abs() > busy_eps).astype(int)  # Nicht mehr nötig

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

        # BESS-Leistungsgrenzen: verfügbare Restkapazität nutzen
        if ignore_busy:
            # Was-wäre-wenn: komplette BESS-Leistung verfügbar
            available_power = P_max
        else:
            # Realität: nur die noch nicht belegte Leistung nutzen
            busy_power = abs(float(df.loc[i, "bess_busy_kw"]))
            available_power = max(0.0, P_max - busy_power)
        
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

# ---------------------- Freier Handel Optimierer (Vergleichsszenario) ----------------------
def optimize_free_trading(df: pd.DataFrame,
                         E_max: float, P_max: float,
                         eta_ch: float, eta_dis: float, dt_h: float,
                         soc0_extra_kwh: float, fix_final: bool,
                         fee_buy: float, fee_sell: float,
                         cycles_cap: float, cycles_constraint: str) -> Optional[pd.DataFrame]:
    """
    Optimierung ohne Netz- und Eigenverbrauch-Constraints.
    Nur Batterie-Parameter und Zyklenlimit.
    """
    if pulp is None:
        return None

    n = len(df)
    if n == 0:
        return None

    # Problem definieren
    prob = pulp.LpProblem("BESS_Free_Trading", pulp.LpMaximize)

    # Variablen
    ch  = pulp.LpVariable.dicts("ch_free_kw",  range(n), lowBound=0, upBound=P_max)
    dis = pulp.LpVariable.dicts("dis_free_kw", range(n), lowBound=0, upBound=P_max)
    soc = pulp.LpVariable.dicts("soc_free_kwh", range(n), lowBound=0, upBound=E_max)

    # Zielfunktion: Erlös aus freiem Handel
    prob += pulp.lpSum(
        ((float(df.loc[i,"price_eur_per_mwh"]) - fee_sell) * dis[i] - 
         (float(df.loc[i,"price_eur_per_mwh"]) + fee_buy) * ch[i]) * (dt_h/1000.0)
        for i in range(n)
    )

    # SoC-Dynamik
    prob += soc[0] == soc0_extra_kwh + (eta_ch*ch[0] - (dis[0]/eta_dis)) * dt_h
    for i in range(1, n):
        prob += soc[i] == soc[i-1] + (eta_ch*ch[i] - (dis[i]/eta_dis)) * dt_h

    # Zyklenlimit (falls aktiviert)
    if cycles_cap and cycles_cap > 0 and E_max > 0 and cycles_constraint != "Ignorieren":
        total_discharge = pulp.lpSum(dis[i] for i in range(n)) * dt_h
        
        if cycles_constraint == "Exakte Ausnutzung":
            prob += total_discharge == cycles_cap * E_max
        else:  # Obergrenze
            prob += total_discharge <= cycles_cap * E_max

    # End-SoC Constraint
    if fix_final:
        prob += soc[n-1] == soc0_extra_kwh

    # Lösen
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[prob.status] != "Optimal":
            return None

    except Exception as e:
        return None

    # Ergebnisse für Vergleich
    result = df[["price_eur_per_mwh"]].copy()
    result["p_ch_free_kw"]  = [pulp.value(ch[i]) or 0.0 for i in range(n)]
    result["p_dis_free_kw"] = [pulp.value(dis[i]) or 0.0 for i in range(n)]
    result["soc_free_kwh"] = [pulp.value(soc[i]) or 0.0 for i in range(n)]
    result["e_ch_free_kwh"]  = result["p_ch_free_kw"]  * dt_h
    result["e_dis_free_kwh"] = result["p_dis_free_kw"] * dt_h
    result["revenue_free_eur"] = ((result["price_eur_per_mwh"] - fee_sell) * result["e_dis_free_kwh"] - 
                                 (result["price_eur_per_mwh"] + fee_buy) * result["e_ch_free_kwh"]) / 1000.0

    return result

# ---------------------- Optimierung ausführen ----------------------
run_button = st.button("🚀 Optimierung starten", type="primary")
if not run_button:
    st.stop()

with st.spinner("Optimierung läuft..."):
    # Hauptoptimierung: mit allen Constraints
    result = optimize_bess(
        DF.copy(), E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
        soc0_extra_kwh, fix_final, fee_buy, fee_sell, 
        cycles_cap, cycles_constraint, ignore_busy
    )
    
    # Vergleichsoptimierung: komplett freier Handel (ohne Netz/Eigenverbrauch-Constraints)
    st.info("Berechne Vergleichsszenario: komplett freier Handel...")
    result_free = optimize_free_trading(
        DF.copy(), E_max, P_max, eta_ch, eta_dis, dt_h,
        soc0_extra_kwh, fix_final, fee_buy, fee_sell, 
        cycles_cap, cycles_constraint
    )

if result is None or result_free is None:
    if result is None:
        st.error("Hauptoptimierung fehlgeschlagen!")
    if result_free is None:
        st.error("Vergleichsoptimierung fehlgeschlagen!")
    st.stop()

# ---------------------- Ergebnisse & KPIs ----------------------
st.subheader("📊 Ergebnisse")

# Hauptoptimierung (mit Constraints)
revenue_total = result["revenue_eur"].sum()
energy_charged = result["e_ch_kwh"].sum()
energy_discharged = result["e_dis_kwh"].sum()
arbitrage_cycles = energy_discharged / E_max if E_max > 0 else 0
baseline_cycles = DF["e_dis_base_kwh"].sum() / E_max if E_max > 0 else 0
total_cycles = arbitrage_cycles + baseline_cycles

# Freier Handel (ohne Constraints)
revenue_free = result_free["revenue_free_eur"].sum()
energy_charged_free = result_free["e_ch_free_kwh"].sum()
energy_discharged_free = result_free["e_dis_free_kwh"].sum()
cycles_free = energy_discharged_free / E_max if E_max > 0 else 0

# Vergleichsmetriken
revenue_gap_abs = revenue_free - revenue_total
revenue_gap_pct = (revenue_gap_abs / revenue_free * 100) if revenue_free != 0 else 0

# Hauptmetriken
st.write("### 🎯 Hauptvergleich: Realistisch vs. Freier Handel")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Erlös (realistisch)", f"{revenue_total:,.0f} €", 
             help="Mit Netz-, SoC- und Eigenverbrauch-Constraints")
with col2:
    st.metric("Erlös (freier Handel)", f"{revenue_free:,.0f} €", 
             help="Nur Batterie-Parameter und Zyklenlimit")
with col3:
    st.metric("Verlust durch Constraints", f"-{revenue_gap_abs:,.0f} € ({revenue_gap_pct:.1f}%)",
             delta=f"{revenue_gap_pct:.1f}%", delta_color="inverse")

# Detailmetriken
st.write("### 📋 Detaillierte Kennzahlen")
col4, col5, col6, col7 = st.columns(4)
with col4:
    st.metric("Arbitrage-Zyklen (real)", f"{arbitrage_cycles:.1f}")
    st.metric("Arbitrage-Zyklen (frei)", f"{cycles_free:.1f}")
with col5:
    st.metric("Gesamt-Zyklen (real)", f"{total_cycles:.1f}")
    st.metric("Nur Arbitrage (frei)", f"{cycles_free:.1f}")
with col6:
    efficiency = (energy_discharged / energy_charged * 100) if energy_charged > 0 else 0
    efficiency_free = (energy_discharged_free / energy_charged_free * 100) if energy_charged_free > 0 else 0
    st.metric("Effizienz (real)", f"{efficiency:.1f}%")
    st.metric("Effizienz (frei)", f"{efficiency_free:.1f}%")
with col7:
    total_capacity = len(result) * P_max * dt_h
    used_capacity = result["p_ch_kw"].sum() + result["p_dis_kw"].sum()
    used_capacity_free = result_free["p_ch_free_kw"].sum() + result_free["p_dis_free_kw"].sum()
    capacity_utilization = (used_capacity / total_capacity * 100) if total_capacity > 0 else 0
    capacity_utilization_free = (used_capacity_free / total_capacity * 100) if total_capacity > 0 else 0
    st.metric("Kapazitätsnutzung (real)", f"{capacity_utilization:.1f}%")
    st.metric("Kapazitätsnutzung (frei)", f"{capacity_utilization_free:.1f}%")

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

# KPI-Zusammenfassung für Export
kpi_summary = pd.DataFrame({
    "Kennzahl": [
        "Erlös realistisch [€]", "Erlös freier Handel [€]", "Verlust absolut [€]", "Verlust relativ [%]",
        "Energie geladen real [kWh]", "Energie geladen frei [kWh]", 
        "Energie entladen real [kWh]", "Energie entladen frei [kWh]",
        "Arbitrage-Zyklen real [#]", "Arbitrage-Zyklen frei [#]",
        "Baseline-Zyklen [#]", "Gesamt-Zyklen real [#]",
        "Roundtrip-Effizienz real [%]", "Roundtrip-Effizienz frei [%]",
        "Ø Lade-Preis real [€/MWh]", "Ø Lade-Preis frei [€/MWh]",
        "Ø Entlade-Preis real [€/MWh]", "Ø Entlade-Preis frei [€/MWh]",
        "Preis-Spread real [€/MWh]", "Preis-Spread frei [€/MWh]",
        "Max. Netzlast [kW]", "Kapazitätsnutzung real [%]", "Kapazitätsnutzung frei [%]",
        "RTE [%]", "E_max [kWh]", "P_max [kW]", "P_conn [kW]", "Zyklenlimit [#/Jahr]"
    ],
    "Wert": [
        round(revenue_total, 2), round(revenue_free, 2), round(revenue_gap_abs, 2), round(revenue_gap_pct, 1),
        round(energy_charged, 1), round(energy_charged_free, 1),
        round(energy_discharged, 1), round(energy_discharged_free, 1),
        round(arbitrage_cycles, 2), round(cycles_free, 2),
        round(baseline_cycles, 2), round(total_cycles, 2),
        round(efficiency, 1), round(efficiency_free, 1),
        round(avg_price_charge, 1), round(avg_price_charge_free, 1),
        round(avg_price_discharge, 1), round(avg_price_discharge_free, 1),
        round(price_spread, 1), round(price_spread_free, 1),
        round(max_net_load, 1), round(capacity_utilization, 1), round(capacity_utilization_free, 1),
        rte_pct, E_max, P_max, P_conn, cycles_cap
    ]
})

# Excel-Export erstellen
bio = BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as writer:
    result.to_excel(writer, index=False, sheet_name="Zeitreihen_Real")
    result_free.to_excel(writer, index=False, sheet_name="Zeitreihen_FreierHandel")
    kpi_summary.to_excel(writer, index=False, sheet_name="KPIs_Vergleich")

bio.seek(0)

st.download_button(
    "📥 Ergebnisse als Excel herunterladen",
    data=bio,
    file_name="BESS_Arbitrage_Ergebnisse.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
