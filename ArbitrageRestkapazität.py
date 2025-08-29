# -*- coding: utf-8 -*-
from io import BytesIO
from typing import Optional, List
import numpy as np
import pandas as pd
import streamlit as st

# Optional: Solver
try:
    import pulp
except Exception:
    pulp = None

# Optional: Charts
import altair as alt

# ------- Robuste Zahleneingaben (sprach-/format-tolerant) -------
# Erlaubt Eingaben wie 180000, 180.000, 180 000, 180.000,5, 180000.5, 90%, 10k, 1M
import re

def _parse_local_float(s, default=None):
    try:
        if s is None:
            return float(default) if default is not None else 0.0
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s).strip().replace(" ", "")
        if s == "":
            return float(default) if default is not None else 0.0
        # Prozentzeichen tolerieren
        if s.endswith("%"):
            s = s[:-1]
        # k/M Suffix
        mult = 1.0
        if s.endswith(("k", "K")):
            mult = 1e3; s = s[:-1]
        elif s.endswith(("m", "M")):
            mult = 1e6; s = s[:-1]
        # Lokale Dezimal-/Tausender-Trennung
        if s.count(",") > 0 and s.count(".") > 0:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(",") > 0 and s.count(".") == 0:
            s = s.replace(",", ".")
        return float(s) * mult
    except Exception:
        return float(default) if default is not None else 0.0

def ui_float(label, default, min_value=None, max_value=None, key=None, help=None, disabled=False):
    raw = st.text_input(label, value=f"{default}", key=key, help=help, disabled=disabled)
    val = _parse_local_float(raw, default)
    # Begrenzen + Feedback
    if min_value is not None and val < min_value:
        st.warning(f"{label}: Wert unter Minimum ({min_value}). Auf Minimum gesetzt.")
        val = min_value
    if max_value is not None and val > max_value:
        st.warning(f"{label}: Wert über Maximum ({max_value}). Auf Maximum gesetzt.")
        val = max_value
    return val

def ui_int(label, default, min_value=None, max_value=None, key=None, help=None, disabled=False):
    raw = st.text_input(label, value=f"{int(default)}", key=key, help=help, disabled=disabled)
    try:
        val = int(round(_parse_local_float(raw, default)))
    except Exception:
        val = int(default)
    if min_value is not None and val < min_value:
        st.warning(f"{label}: Wert unter Minimum ({min_value}). Auf Minimum gesetzt.")
        val = min_value
    if max_value is not None and val > max_value:
        st.warning(f"{label}: Wert über Maximum ({max_value}). Auf Maximum gesetzt.")
        val = max_value
    return val

st.set_page_config(page_title="BESS – Arbitrage (PV-First)", layout="wide")
st.title("⚡ BESS-Arbitrage mit PV-First-Strategie & intelligenter Restkapazitätsnutzung")

# =============================================================
# ---------------------- Uploads -------------------------------
# =============================================================
cu1, cu2 = st.columns(2)
with cu1:
    base_file = st.file_uploader(
        "Basistabelle (SoC_kWh, Netload_base_kW, BESS_busy_kW, optional PV_generation_kW, optional Zeit)",
        type=["xlsx","xls","csv"], key="base")
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

# =============================================================
# ---------------------- Column mapping -----------------------
# =============================================================
st.subheader("Spaltenzuordnung")
base_cols  = list(base_raw.columns)
price_cols = list(price_raw.columns)

# Robuste Spaltenerkennung

def guess(colnames: List[str], keys: List[str]):
    low = {c.lower(): c for c in colnames}
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return colnames[0] if colnames else None

# Zeitstempel-Erkennung in Basistabelle
has_time_in_base = any(k in " ".join([c.lower() for c in base_cols]) for k in ["zeit","time","date","timestamp"])

# NEU: Moduswahl – PV separat oder in Netzlast saldiert?
pv_in_netload = st.checkbox(
    "Keine separate PV-Spalte (PV bereits in Netzlast saldiert)",
    value=True,
    help=(
        "Aktivieren, wenn es KEINE eigene PV-Spalte gibt. Dann wird der PV-Überschuss aus der negativen Netzlast "
        "(inkl. BESS_busy) hergeleitet: pv_surplus = max(0, -(netload_base + bess_busy))."
    )
)

b_time = None
if has_time_in_base:
    guess_btime = guess(base_cols,["zeit","time","date"]) if base_cols else None
    b_time = st.selectbox(
        "Basistabelle: Zeitstempel (optional)",
        [None] + base_cols,
        index=(base_cols.index(guess_btime) + 1) if (guess_btime and guess_btime in base_cols) else 0
    )

b_soc   = st.selectbox("Basistabelle: SoC_kWh (Baseline)", base_cols,
                       index=base_cols.index(guess(base_cols,["soc"])) if guess(base_cols,["soc"]) else 0)
b_nload = st.selectbox("Basistabelle: belegte Netzleistung [kW] (±)", base_cols,
                       index=base_cols.index(guess(base_cols,["net","last","grid","anschluss"])) if guess(base_cols,["net","last","grid","anschluss"]) else 0)
b_busyP = st.selectbox("Basistabelle: BESS_busy_kW (±)", base_cols,
                       index=base_cols.index(guess(base_cols,["bess","busy","lade","entlade","power"])) if guess(base_cols,["bess","busy","lade","entlade","power"]) else 0)

# PV-Spalte nur abfragen, wenn nicht in Netzlast saldiert
if not pv_in_netload:
    b_pv = st.selectbox("Basistabelle: PV_generation_kW (+)", base_cols,
                        index=base_cols.index(guess(base_cols,["pv","solar","generation","erzeugung"])) if guess(base_cols,["pv","solar","generation","erzeugung"]) else 0)
else:
    b_pv = None

p_time = st.selectbox("Preise: Zeitstempel", price_cols,
                      index=price_cols.index(guess(price_cols,["zeit","time","date"])) if guess(price_cols,["zeit","time","date"]) else 0)
p_val  = st.selectbox("Preise: Preis-Spalte", price_cols,
                      index=price_cols.index(guess(price_cols,["preis","price","eur","ct"])) if guess(price_cols,["preis","price","eur","ct"]) else 0)

price_unit = st.radio("Preiseinheit", ["€/MWh","€/kWh","ct/kWh"], horizontal=True)

# =============================================================
# --------------- Batterie- & Netz-Parameter ------------------
# =============================================================
st.subheader("Batterie- & Netz-Parameter")
cp1, cp2, cp3 = st.columns(3)
with cp1:
    E_max = ui_float("E_max nutzbar [kWh]", 150.0, min_value=0.0, key="E_max")
with cp2:
    P_max = ui_float("BESS P_max (sym) [kW]", 100.0, min_value=0.0, key="P_max")
with cp3:
    P_conn = ui_float("Netzanschluss P_conn [kW] (symmetrisch)", 73.0, min_value=0.0, key="P_conn", help="Max. absolute Netzleistung |P| ≤ P_conn")

if P_max > P_conn:
    st.warning("⚠️ BESS-Leistung > Netzanschluss - kann zu suboptimalen Lösungen führen")

cp4, cp5, cp6 = st.columns(3)
with cp4:
    rte_pct = ui_float("RTE [%]", 95.0, min_value=0.0, max_value=100.0, key="RTE", help="Round-Trip-Effizienz in %. Beispiele: 90, 90%. Bitte als Prozent eingeben.")
with cp5:
    soc0_extra_pct = st.slider("Start-SoC_extra [% von E_max]", min_value=0, max_value=100, value=0)
with cp6:
    fix_final = st.checkbox("Finaler SoC_extra = Start-SoC_extra", value=True)

# Soft-Validierung für RTE (<50% oft unrealistisch)
if rte_pct < 50.0:
    st.warning("⚠️ Sehr niedrige RTE (<50%) eingegeben – bitte prüfen. Üblich sind 80–95%.")

cp7, cp8, cp9 = st.columns(3)
with cp7:
    cycles_cap = ui_float(
        "Max. Gesamt-Vollzyklen/Jahr [#]",
        0.0, min_value=0.0, key="cycles_cap",
        help="0 = keine Begrenzung; gilt für Baseline + Zusatz"
    )
with cp8:
    fees = ui_float("Gebühren [€/MWh] (Kauf/Verkauf)", 0.0, min_value=0.0, key="fees")
with cp9:
    enable_eeg = st.checkbox("EEG-Vergütung aktiv", value=True,
                             help="Wenn aktiv: PV-Einspeisewert je Slot = 0 bei negativen Preisen, sonst max(EEG, Day-Ahead)")
    eeg_tariff = ui_float("EEG-Vergütungssatz [€/MWh]", 80.0, min_value=0.0, key="eeg_tariff")

cp10, cp11 = st.columns(2)
with cp10:
    cycles_constraint = st.radio("Zyklen-Constraint", ["Obergrenze", "Exakte Ausnutzung", "Ignorieren"],
                                  horizontal=False,
                                 help="Obergrenze: ≤ Limit, Exakte Ausnutzung: = Limit")
with cp11:
    ignore_busy = st.checkbox("BESS_busy ignorieren (nur Test)", value=False)

# Parameter berechnen
eta_rt  = rte_pct/100.0
eta_ch  = eta_rt ** 0.5
eta_dis = eta_rt ** 0.5
soc0_extra_kwh = E_max * (soc0_extra_pct/100.0)
fee_buy = float(fees)
fee_sell = float(fees)

# =============================================================
# ---------------------- EEG-Vergütungslogik ------------------
# =============================================================

def calculate_eeg_feed_in_value(price_eur_mwh: float, eeg_tariff: float, enable_eeg: bool) -> float:
    """
    Effektiver PV-Einspeisewert pro Slot (vereinfachtes EEG-Marktprämienmodell):
      - Negative Preise: 0
      - 0 ≤ Preis < EEG: EEG
      - Preis ≥ EEG: Preis
    Wenn EEG deaktiviert: max(0, Preis)
    """
    if not enable_eeg:
        return max(0.0, float(price_eur_mwh))
    p = float(price_eur_mwh)
    if p < 0:
        return 0.0
    elif p < eeg_tariff:
        return float(eeg_tariff)
    else:
        return p

# =============================================================
# ----------------- Prognoseunsicherheiten (MC) ---------------
# =============================================================

with st.expander("🎲 Prognoseunsicherheit (Monte-Carlo)"):
    uncertainty_analysis = st.checkbox("Unsicherheitsanalyse aktivieren", value=False)
    colu1, colu2, colu3 = st.columns(3)
    with colu1:
        n_scenarios = ui_int("Szenarien [#]", 50, min_value=2, key="n_scenarios", disabled=not uncertainty_analysis)
    with colu2:
        pv_uncertainty = ui_float("PV-Prognosefehler σ [%]", 10.0, min_value=0.0, key="pv_sigma", disabled=not uncertainty_analysis)
    with colu3:
        price_uncertainty = ui_float("Preis-Prognosefehler σ [%]", 20.0, min_value=0.0, key="price_sigma", disabled=not uncertainty_analysis)
    correlation = st.slider("Korrelation PV↔Preis", min_value=-1.0, max_value=1.0, value=-0.2, step=0.05, disabled=not uncertainty_analysis)

def generate_uncertainty_scenarios(df: pd.DataFrame, n_scenarios: int,
                                   pv_uncertainty: float, price_uncertainty: float,
                                   correlation: float) -> list:
    """Erzeugt korrelierte Störszenarien für PV und Preise."""
    scenarios = []
    n_points = len(df)
    cov_matrix = np.array([[pv_uncertainty**2, correlation * pv_uncertainty * price_uncertainty],
                           [correlation * pv_uncertainty * price_uncertainty, price_uncertainty**2]])
    for _ in range(n_scenarios):
        errors = np.random.multivariate_normal([0, 0], cov_matrix, n_points)
        pv_errors = errors[:, 0] / 100.0
        price_errors = errors[:, 1] / 100.0
        scenario_df = df.copy()
        # Bei saldierter PV gibt es keine PV-Generation, aber pv_surplus_inferred_kw bleibt gültig
        if "pv_generation_kw" in df.columns:
            scenario_df["pv_generation_kw"] = np.maximum(0.0, df.get("pv_generation_kw", 0.0) * (1.0 + pv_errors))
        if "pv_surplus_inferred_kw" in df.columns:
            scenario_df["pv_surplus_inferred_kw"] = np.maximum(0.0, df["pv_surplus_inferred_kw"] * (1.0 + pv_errors))
        scenario_df["price_eur_per_mwh"] = df["price_eur_per_mwh"] * (1.0 + price_errors)
        scenarios.append(scenario_df)
    return scenarios

# =============================================================
# ---------------------- Datenaufbereitung --------------------
# =============================================================
# Basistabelle
cols = [b_soc, b_nload, b_busyP]
if not pv_in_netload and b_pv is not None:
    cols.append(b_pv)
if b_time:
    cols = [b_time] + cols
base = base_raw[cols].copy()
expected_names = ["soc_base_kwh","netload_base_kw","bess_busy_kw"] + ([] if pv_in_netload else ["pv_generation_kw"])
base.columns = (["ts"] if b_time else []) + expected_names

# Wenn PV saldiert ist, Platzhalter-Spalte anlegen (für Kompatibilität)
if pv_in_netload:
    base["pv_generation_kw"] = 0.0

# Datenvalidierung und -bereinigung
try:
    if b_time:
        base["ts"] = pd.to_datetime(base["ts"], errors="coerce")
        if base["ts"].isna().any():
            st.error("Zeitstempel in Basistabelle konnten nicht konvertiert werden!")
            st.stop()
        base = base.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    base["soc_base_kwh"]      = pd.to_numeric(base["soc_base_kwh"], errors="coerce").fillna(0.0)
    base["netload_base_kw"]   = pd.to_numeric(base["netload_base_kw"], errors="coerce").fillna(0.0)
    base["bess_busy_kw"]      = pd.to_numeric(base["bess_busy_kw"], errors="coerce").fillna(0.0)
    base["pv_generation_kw"]  = pd.to_numeric(base.get("pv_generation_kw", 0.0), errors="coerce").fillna(0.0)

    if (base["soc_base_kwh"] > E_max).any():
        st.error("❌ SoC_baseline übersteigt E_max in manchen Slots!")
        st.stop()
    if (base["soc_base_kwh"] < 0).any():
        st.warning("⚠️ Negative SoC_baseline-Werte gefunden - werden auf 0 gesetzt")
        base["soc_base_kwh"] = base["soc_base_kwh"].clip(lower=0)
    if (base["pv_generation_kw"] < 0).any():
        st.warning("⚠️ Negative PV-Erzeugung gefunden - werden auf 0 gesetzt")
        base["pv_generation_kw"] = base["pv_generation_kw"].clip(lower=0)
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

    # Einheit konvertieren → €/MWh
    if price_unit == "€/MWh":
        price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce")
    elif price_unit == "€/kWh":
        price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 1000.0
    else:  # ct/kWh
        price["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce") * 10.0

    if (price["price_eur_per_mwh"] < -1000).any() or (price["price_eur_per_mwh"] > 1000).any():
        st.warning("⚠️ Ungewöhnliche Preise erkannt – bitte Einheit prüfen")
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
    if len(base) != len(price):
        st.error(f"❌ Ohne Zeitstempel müssen Basistabelle ({len(base)}) und Preise ({len(price)}) gleich lang sein.")
        st.stop()
    DF = base.copy()
    DF["price_eur_per_mwh"] = pd.to_numeric(price["price_raw"], errors="coerce").values
    dt_h = ui_float("Zeitschrittlänge [h]", 0.25, min_value=0.01, max_value=24.0, key="dt_h")

if len(DF) > 8760:
    st.warning(f"⚠️ Große Datenmenge ({len(DF)} Zeitschritte) – Optimierung kann dauern")

# Baseline-Energie
DF["e_dis_base_kwh"] = np.maximum(DF["bess_busy_kw"], 0.0) * dt_h
DF["e_ch_base_kwh"]  = np.maximum(-DF["bess_busy_kw"], 0.0) * dt_h

# NEU: PV-Überschuss vorab herleiten, falls PV in Netzlast saldiert ist
if pv_in_netload:
    DF["base_net_kw"] = DF["netload_base_kw"] + DF["bess_busy_kw"]
    DF["pv_surplus_inferred_kw"] = np.maximum(0.0, -DF["base_net_kw"])  # Export => verfügbarer PV-Überschuss

# Zyklenlimit-Vorprüfung
if cycles_cap and cycles_cap > 0 and E_max > 0:
    base_cycles_total = DF["e_dis_base_kwh"].sum() / E_max
    if base_cycles_total > cycles_cap:
        st.error(f"❌ Zyklenlimit bereits durch Baseline überschritten: {base_cycles_total:.2f} > {cycles_cap}. Keine Arbitrage möglich.")
        if not ignore_busy:
            st.stop()

# =============================================================
# ---------------- PV-First BESS Optimierer -------------------
# =============================================================

def optimize_bess_pv_first(df: pd.DataFrame,
                          E_max: float, P_max: float, P_conn: float,
                          eta_ch: float, eta_dis: float, dt_h: float,
                          soc0_extra_kwh: float, fix_final: bool,
                          fee_buy: float, fee_sell: float,
                          eeg_tariff: float, enable_eeg: bool,
                          cycles_cap: float, cycles_constraint: str,
                          ignore_busy: bool = False) -> Optional[pd.DataFrame]:

    if pulp is None:
        st.error("❌ PuLP nicht verfügbar. Bitte installieren: pip install pulp")
        return None

    n = len(df)
    if n == 0:
        st.error("Keine Daten für Optimierung verfügbar!")
        return None

    prob = pulp.LpProblem("BESS_PV_First_Arbitrage", pulp.LpMaximize)

    # Variablen
    ch_pv   = pulp.LpVariable.dicts("ch_pv_kw", range(n), lowBound=0)
    ch_grid = pulp.LpVariable.dicts("ch_grid_kw", range(n), lowBound=0)
    dis     = pulp.LpVariable.dicts("dis_kw", range(n), lowBound=0)
    soc     = pulp.LpVariable.dicts("soc_extra_kwh", range(n), lowBound=0)

    # PV-Überschuss je Slot bestimmen
    pv_surplus = []
    use_inferred = "pv_surplus_inferred_kw" in df.columns
    for i in range(n):
        if use_inferred:
            surplus = max(0.0, float(df.loc[i, "pv_surplus_inferred_kw"]))
        else:
            pv_gen = float(df.loc[i, "pv_generation_kw"])  # >=0
            consumption_base = abs(min(0.0, float(df.loc[i, "netload_base_kw"])))  # nur Bezug
            bess_busy_consumption = abs(min(0.0, float(df.loc[i, "bess_busy_kw"])))
            surplus = max(0.0, pv_gen - consumption_base - bess_busy_consumption)
        pv_surplus.append(surplus)

    # Zielfunktion
    prob += pulp.lpSum([
        ((float(df.loc[i, "price_eur_per_mwh"]) - fee_sell) * dis[i]
        - (float(df.loc[i, "price_eur_per_mwh"]) + fee_buy) * ch_grid[i]) * (dt_h/1000.0)
        + calculate_eeg_feed_in_value(float(df.loc[i, "price_eur_per_mwh"]), eeg_tariff, enable_eeg) * ch_pv[i] * (dt_h/1000.0)
        for i in range(n)
    ])

    # SoC-Dynamik
    prob += soc[0] == soc0_extra_kwh + (eta_ch*(ch_pv[0] + ch_grid[0]) - (dis[0]/eta_dis)) * dt_h
    for i in range(1, n):
        prob += soc[i] == soc[i-1] + (eta_ch*(ch_pv[i] + ch_grid[i]) - (dis[i]/eta_dis)) * dt_h

    # Constraints je Slot
    for i in range(n):
        # (1) PV-Laden begrenzen
        prob += ch_pv[i] <= pv_surplus[i]

        # (2) SoC-Headroom
        soc_base_i = float(df.loc[i, "soc_base_kwh"])  # Baseline-SoC belegt Speicher
        headroom = max(0.0, E_max - soc_base_i)
        prob += soc[i] <= headroom

        # (3) BESS-Leistungsgrenzen
        if ignore_busy:
            available_power = P_max
        else:
            busy_power = abs(float(df.loc[i, "bess_busy_kw"]))
            available_power = max(0.0, P_max - busy_power)
        prob += ch_pv[i] + ch_grid[i] <= available_power
        prob += dis[i] <= available_power

        # (4) Netzanschluss (PV-Laden belastet Netz nicht)
        base_net = float(df.loc[i, "netload_base_kw"]) + float(df.loc[i, "bess_busy_kw"])  # kann ± sein
        net_total = base_net + (dis[i] - ch_grid[i])
        prob += net_total <= P_conn
        prob += net_total >= -P_conn

    # (5) Zyklenlimit
    if cycles_cap and cycles_cap > 0 and E_max > 0 and cycles_constraint != "Ignorieren":
        base_cycles = float(df["e_dis_base_kwh"].sum()) if not ignore_busy else 0.0
        additional_discharge = pulp.lpSum(dis[i] for i in range(n)) * dt_h
        if cycles_constraint == "Exakte Ausnutzung":
            prob += additional_discharge + base_cycles == cycles_cap * E_max
        else:
            prob += additional_discharge + base_cycles <= cycles_cap * E_max

    # (6) End-SoC
    if fix_final:
        prob += soc[n-1] == soc0_extra_kwh

    # Solve
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != "Optimal":
            st.error(f"❌ PV-First-Optimierung fehlgeschlagen: {pulp.LpStatus[prob.status]}")
            return None
    except Exception as e:
        st.error(f"❌ Fehler bei PV-First-Optimierung: {str(e)}")
        return None

    # Ergebnisse
    result = df.copy()
    result["p_ch_pv_kw"]   = [pulp.value(ch_pv[i]) or 0.0 for i in range(n)]
    result["p_ch_grid_kw"] = [pulp.value(ch_grid[i]) or 0.0 for i in range(n)]
    result["p_ch_total_kw"] = result["p_ch_pv_kw"] + result["p_ch_grid_kw"]
    result["p_dis_kw"] = [pulp.value(dis[i]) or 0.0 for i in range(n)]
    result["soc_extra_kwh"] = [pulp.value(soc[i]) or 0.0 for i in range(n)]

    # PV-Bilanzierung
    result["pv_surplus_available_kw"] = pv_surplus
    result["pv_surplus_used_kw"] = result["p_ch_pv_kw"]
    result["pv_surplus_remaining_kw"] = result["pv_surplus_available_kw"] - result["pv_surplus_used_kw"]
    result["pv_curtailment_kw"] = np.maximum(0.0, result["pv_surplus_remaining_kw"])

    # Energie
    result["soc_total_kwh"] = result["soc_base_kwh"] + result["soc_extra_kwh"]
    result["e_ch_pv_kwh"]   = result["p_ch_pv_kw"] * dt_h
    result["e_ch_grid_kwh"] = result["p_ch_grid_kw"] * dt_h
    result["e_ch_total_kwh"] = result["e_ch_pv_kwh"] + result["e_ch_grid_kwh"]
    result["e_dis_kwh"] = result["p_dis_kw"] * dt_h

    # Netzlast (ohne PV-Laden)
    result["net_total_kw"] = (result["netload_base_kw"] + result["bess_busy_kw"] +
                               result["p_dis_kw"] - result["p_ch_grid_kw"])  # PV-Laden zählt nicht aufs Netz

    # Wirtschaftlichkeit
    result["revenue_arbitrage_eur"] = (
        (result["price_eur_per_mwh"] - fee_sell) * result["e_dis_kwh"] -
        (result["price_eur_per_mwh"] + fee_buy) * result["e_ch_grid_kwh"]
    ) / 1000.0

    result["eeg_feed_in_value_eur_mwh"] = result["price_eur_per_mwh"].apply(
        lambda x: calculate_eeg_feed_in_value(x, eeg_tariff, enable_eeg))

    result["pv_value_used_eur"] = result["e_ch_pv_kwh"] * result["eeg_feed_in_value_eur_mwh"] / 1000.0
    result["pv_curtailment_avoided_value_eur"] = result["pv_surplus_used_kw"] * dt_h * result["eeg_feed_in_value_eur_mwh"] / 1000.0

    result["revenue_total_eur"] = result["revenue_arbitrage_eur"] + result["pv_value_used_eur"]

    return result

# =============================================================
# ----------- Vergleich: komplett freier Handel ----------------
# =============================================================

def optimize_free_trading(df: pd.DataFrame,
                         E_max: float, P_max: float,
                         eta_ch: float, eta_dis: float, dt_h: float,
                         soc0_extra_kwh: float, fix_final: bool,
                         fee_buy: float, fee_sell: float,
                         cycles_cap: float, cycles_constraint: str) -> Optional[pd.DataFrame]:
    if pulp is None:
        return None
    n = len(df)
    if n == 0:
        return None

    prob = pulp.LpProblem("BESS_Free_Trading", pulp.LpMaximize)

    ch  = pulp.LpVariable.dicts("ch_free_kw",  range(n), lowBound=0, upBound=P_max)
    dis = pulp.LpVariable.dicts("dis_free_kw", range(n), lowBound=0, upBound=P_max)
    soc = pulp.LpVariable.dicts("soc_free_kwh", range(n), lowBound=0, upBound=E_max)

    prob += pulp.lpSum(
        ((float(df.loc[i,"price_eur_per_mwh"]) - fee_sell) * dis[i] -
          (float(df.loc[i,"price_eur_per_mwh"]) + fee_buy) * ch[i]) * (dt_h/1000.0)
        for i in range(n)
    )

    prob += soc[0] == soc0_extra_kwh + (eta_ch*ch[0] - (dis[0]/eta_dis)) * dt_h
    for i in range(1, n):
        prob += soc[i] == soc[i-1] + (eta_ch*ch[i] - (dis[i]/eta_dis)) * dt_h

    if cycles_cap and cycles_cap > 0 and E_max > 0 and cycles_constraint != "Ignorieren":
        total_discharge = pulp.lpSum(dis[i] for i in range(n)) * dt_h
        if cycles_constraint == "Exakte Ausnutzung":
            prob += total_discharge == cycles_cap * E_max
        else:
            prob += total_discharge <= cycles_cap * E_max

    if fix_final:
        prob += soc[n-1] == soc0_extra_kwh

    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != "Optimal":
            return None
    except Exception:
        return None

    result = df[["price_eur_per_mwh"]].copy()
    result["p_ch_free_kw"]  = [pulp.value(ch[i]) or 0.0 for i in range(n)]
    result["p_dis_free_kw"] = [pulp.value(dis[i]) or 0.0 for i in range(n)]
    result["soc_free_kwh"] = [pulp.value(soc[i]) or 0.0 for i in range(n)]
    result["e_ch_free_kwh"]  = result["p_ch_free_kw"]  * dt_h
    result["e_dis_free_kwh"] = result["p_dis_free_kw"] * dt_h
    result["revenue_free_eur"] = (
        (result["price_eur_per_mwh"] - fee_sell) * result["e_dis_free_kwh"] - 
         (result["price_eur_per_mwh"] + fee_buy) * result["e_ch_free_kwh"]
    ) / 1000.0
    return result

# =============================================================
# ---------------------- Optimierung starten ------------------
# =============================================================
run_button = st.button("🚀 Optimierung starten", type="primary")
if not run_button:
    st.stop()

with st.spinner("Optimierung läuft..."):
    if uncertainty_analysis and n_scenarios and n_scenarios > 1:
        st.info(f"Führe Monte-Carlo-Simulation mit {int(n_scenarios)} Szenarien durch…")
        scenarios = generate_uncertainty_scenarios(DF, int(n_scenarios), float(pv_uncertainty), float(price_uncertainty), float(correlation))
        scenario_revenues = []
        progress_bar = st.progress(0)
        for i, scen_df in enumerate(scenarios):
            scen_res = optimize_bess_pv_first(
                scen_df, E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
                soc0_extra_kwh, fix_final, fee_buy, fee_sell, eeg_tariff, enable_eeg,
                cycles_cap, cycles_constraint, ignore_busy
            )
            if scen_res is not None:
                scenario_revenues.append(float(scen_res["revenue_total_eur"].sum()))
            progress_bar.progress((i + 1) / int(n_scenarios))

        # Hauptoptimierung (Deterministisch)
        result = optimize_bess_pv_first(
            DF.copy(), E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
            soc0_extra_kwh, fix_final, fee_buy, fee_sell, eeg_tariff, enable_eeg,
            cycles_cap, cycles_constraint, ignore_busy
        )

        # Unsicherheits-Kennzahlen
        if len(scenario_revenues) > 0:
            uncertainty_stats = {
                "mean": float(np.mean(scenario_revenues)),
                "std": float(np.std(scenario_revenues)),
                "min": float(np.min(scenario_revenues)),
                "max": float(np.max(scenario_revenues)),
                "p5":  float(np.percentile(scenario_revenues, 5)),
                "p95": float(np.percentile(scenario_revenues, 95)),
                "scenarios": scenario_revenues
            }
        else:
            uncertainty_stats = None
            st.error("Monte-Carlo-Simulation fehlgeschlagen!")
    else:
        # Deterministisch
        result = optimize_bess_pv_first(
            DF.copy(), E_max, P_max, P_conn, eta_ch, eta_dis, dt_h,
            soc0_extra_kwh, fix_final, fee_buy, fee_sell, eeg_tariff, enable_eeg,
            cycles_cap, cycles_constraint, ignore_busy
        )
        uncertainty_stats = None

    # Vergleich (frei)
    st.info("Berechne Vergleichsszenario: komplett freier Handel…")
    result_free = optimize_free_trading(
        DF.copy(), E_max, P_max, eta_ch, eta_dis, dt_h,
        soc0_extra_kwh, fix_final, fee_buy, fee_sell, 
        cycles_cap, cycles_constraint
    )

if result is None or result_free is None:
    if result is None:
        st.error("PV-First-Optimierung fehlgeschlagen!")
    if result_free is None:
        st.error("Vergleichsoptimierung fehlgeschlagen!")
    st.stop()

# =============================================================
# ---------------------- Unsicherheits-UI ---------------------
# =============================================================
if uncertainty_analysis and uncertainty_stats is not None:
    st.subheader("🎲 Prognoseunsicherheit (Monte-Carlo-Simulation)")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    with col_u1:
        st.metric("Erwartungswert", f"{uncertainty_stats['mean']:,.0f} €")
        st.metric("Standardabweichung", f"{uncertainty_stats['std']:,.0f} €")
    with col_u2:
        st.metric("Minimum (worst case)", f"{uncertainty_stats['min']:,.0f} €")
        st.metric("5%-Quantil", f"{uncertainty_stats['p5']:,.0f} €")
    with col_u3:
        st.metric("Maximum (best case)", f"{uncertainty_stats['max']:,.0f} €")
        st.metric("95%-Quantil", f"{uncertainty_stats['p95']:,.0f} €")
    with col_u4:
        confidence_90 = uncertainty_stats['p95'] - uncertainty_stats['p5']
        risk_metric = (uncertainty_stats['std'] / uncertainty_stats['mean'] * 100) if uncertainty_stats['mean'] > 0 else 0
        st.metric("90%-Konfidenzintervall", f"{confidence_90:,.0f} €")
        st.metric("Variationskoeffizient", f"{risk_metric:.1f}%")

    scenario_data = pd.DataFrame({"Erlös": uncertainty_stats['scenarios']})
    hist_chart = alt.Chart(scenario_data).mark_bar().encode(
        alt.X("Erlös:Q", bin=alt.Bin(maxbins=30), title="Erlös [€]"),
        alt.Y("count()", title="Häufigkeit"),
    ).properties(title="Verteilung der Erlöse (Monte-Carlo-Simulation)", height=200)

    quantile_data = pd.DataFrame({
        "Quantil": ["5%", "50%", "95%"],
        "Wert": [uncertainty_stats['p5'], uncertainty_stats['mean'], uncertainty_stats['p95']]
    })
    lines_chart = alt.Chart(quantile_data).mark_rule(color='red', strokeDash=[5,5]).encode(x='Wert:Q', color=alt.Color('Quantil:N'))
    st.altair_chart(hist_chart + lines_chart, use_container_width=True)

# =============================================================
# ---------------------- KPI-Berechnung -----------------------
# =============================================================

def calculate_pv_kpis(result: pd.DataFrame, dt_h: float, E_max: float) -> dict:
    # Energiebilanzen
    e_ch_pv_total = result["e_ch_pv_kwh"].sum()
    e_ch_grid_total = result["e_ch_grid_kwh"].sum()
    e_ch_total = e_ch_pv_total + e_ch_grid_total
    e_dis_total = result["e_dis_kwh"].sum()

    # PV-Nutzung
    pv_surplus_total = result["pv_surplus_available_kw"].sum() * dt_h
    pv_used_total = result["pv_surplus_used_kw"].sum() * dt_h
    pv_curtailed_total = result["pv_curtailment_kw"].sum() * dt_h

    pv_utilization_rate = (pv_used_total / pv_surplus_total * 100.0) if pv_surplus_total > 0 else 0.0
    pv_share_in_charging = (e_ch_pv_total / e_ch_total * 100.0) if e_ch_total > 0 else 0.0

    # Zyklen
    pv_cycles = e_ch_pv_total / E_max if E_max > 0 else 0.0
    grid_cycles = e_ch_grid_total / E_max if E_max > 0 else 0.0
    total_cycles = e_dis_total / E_max if E_max > 0 else 0.0

    # Wirtschaftlichkeit
    arbitrage_revenue = float(result["revenue_arbitrage_eur"].sum())
    pv_value = float(result["pv_value_used_eur"].sum())
    total_revenue = float(result["revenue_total_eur"].sum())

    return {
        "pv_surplus_total": pv_surplus_total,
        "pv_used_total": pv_used_total,
        "pv_curtailed_total": pv_curtailed_total,
        "pv_utilization_rate": pv_utilization_rate,
        "pv_share_in_charging": pv_share_in_charging,
        "pv_cycles": pv_cycles,
        "grid_cycles": grid_cycles,
        "total_cycles": total_cycles,
        "arbitrage_revenue": arbitrage_revenue,
        "pv_value": pv_value,
        "total_revenue": total_revenue,
        "e_ch_pv_total": e_ch_pv_total,
        "e_ch_grid_total": e_ch_grid_total,
        "e_dis_total": e_dis_total
    }

st.subheader("📊 Ergebnisse")

pv_kpis = calculate_pv_kpis(result, dt_h, E_max)

# EEG: energiegewichteter Einspeisewert für genutzte PV & negative-Preis-Stunden (früh berechnet)
if pv_kpis["e_ch_pv_total"] > 0:
    eeg_value_weighted = (
        (result["eeg_feed_in_value_eur_mwh"] * result["e_ch_pv_kwh"]).sum() / pv_kpis["e_ch_pv_total"]
    )
else:
    eeg_value_weighted = 0.0
neg_price_hours = int((result["price_eur_per_mwh"] < 0).sum())

# --- PV-Shift vs EEG: Fluss-Tracing (ex-post) ---
soc_pv = 0.0
soc_grid = 0.0
e_dis_pv_kwh_series = []
e_dis_grid_kwh_series = []
for i, row in result.iterrows():
    # in gespeicherter Energie (kWh im Akku)
    soc_pv += eta_ch * float(row["e_ch_pv_kwh"])  # Zuwachs aus PV-Laden
    soc_grid += eta_ch * float(row["e_ch_grid_kwh"])  # Zuwachs aus Grid-Laden

    need_stored = float(row["e_dis_kwh"]) / eta_dis  # benötigte gespeicherte Energie für diese Entladung
    dis_pv_stored = min(soc_pv, need_stored)
    dis_grid_stored = max(0.0, need_stored - dis_pv_stored)

    soc_pv -= dis_pv_stored
    soc_grid -= dis_grid_stored

    e_dis_pv_kwh_series.append(dis_pv_stored * eta_dis)
    e_dis_grid_kwh_series.append(dis_grid_stored * eta_dis)

result["e_dis_pv_kwh"] = e_dis_pv_kwh_series
result["e_dis_grid_kwh"] = e_dis_grid_kwh_series

# Komponenten-Erlöse (mit Gebühren)
revenue_sell_pv_eur = ((result["price_eur_per_mwh"] - fee_sell) * result["e_dis_pv_kwh"]).sum() / 1000.0
revenue_sell_grid_eur = ((result["price_eur_per_mwh"] - fee_sell) * result["e_dis_grid_kwh"]).sum() / 1000.0
cost_buy_grid_eur = ((result["price_eur_per_mwh"] + fee_buy) * result["e_ch_grid_kwh"]).sum() / 1000.0

# EEG-Verlust durch BESS (Opportunitätskosten)
eeg_loss_eur = float(result["pv_value_used_eur"].sum())

# EEG-Export-Baseline vs. mit BESS
baseline_eeg_export_eur = (result["eeg_feed_in_value_eur_mwh"] * result["pv_surplus_available_kw"] * dt_h).sum() / 1000.0
with_bess_eeg_export_eur = (result["eeg_feed_in_value_eur_mwh"] * result["pv_surplus_remaining_kw"] * dt_h).sum() / 1000.0

# Netto-Erlös ggü. EEG-Baseline
net_uplift_vs_eeg_eur = (revenue_sell_pv_eur - eeg_loss_eur) + (revenue_sell_grid_eur - cost_buy_grid_eur)

# Prozent-KPIs ggü. EEG-Baseline (0-Schutz)
if baseline_eeg_export_eur > 0:
    eeg_loss_pct_baseline = 100.0 * eeg_loss_eur / baseline_eeg_export_eur
    eeg_export_reduction_pct = 100.0 * (baseline_eeg_export_eur - with_bess_eeg_export_eur) / baseline_eeg_export_eur
    net_uplift_vs_eeg_pct = 100.0 * net_uplift_vs_eeg_eur / baseline_eeg_export_eur
else:
    eeg_loss_pct_baseline = 0.0
    eeg_export_reduction_pct = 0.0
    net_uplift_vs_eeg_pct = 0.0

# Ø Entladepreis (nur PV-Anteil) und Break-even-Preis
sum_dis_pv = float(result["e_dis_pv_kwh"].sum())
avg_p_dis_pv = (
    (result.loc[result["e_dis_pv_kwh"] > 0, "price_eur_per_mwh"] * result.loc[result["e_dis_pv_kwh"] > 0, "e_dis_pv_kwh"]).sum() / sum_dis_pv
    if sum_dis_pv > 0 else 0.0
)
break_even_price_avg = (eeg_value_weighted / eta_rt) + fee_sell if eta_rt > 0 else float("inf")

# Freier Handel (Vergleich)
revenue_free = float(result_free["revenue_free_eur"].sum())
energy_charged_free = float(result_free["e_ch_free_kwh"].sum())
energy_discharged_free = float(result_free["e_dis_free_kwh"].sum())
cycles_free = energy_discharged_free / E_max if E_max > 0 else 0.0

# Baseline-Zyklen
baseline_cycles = DF["e_dis_base_kwh"].sum() / E_max if E_max > 0 else 0.0
total_cycles_real = pv_kpis["total_cycles"] + baseline_cycles

# Vergleichsmetriken
revenue_gap_abs = revenue_free - pv_kpis["total_revenue"]
revenue_gap_pct = (revenue_gap_abs / revenue_free * 100.0) if revenue_free != 0 else 0.0

# Hauptvergleich
st.write("### 🎯 Hauptvergleich: PV-First vs. Freier Handel")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Erlös (PV-First)", f"{pv_kpis['total_revenue']:,.0f} €",
              help="Mit PV-First-Strategie, Netz- und Eigenverbrauch-Constraints")
with col2:
    st.metric("Erlös (freier Handel)", f"{revenue_free:,.0f} €",
              help="Nur Batterie-Parameter und Zyklenlimit")
with col3:
    st.metric("Verlust durch Constraints", f"-{revenue_gap_abs:,.0f} € ({revenue_gap_pct:.1f}%)",
             delta=f"{revenue_gap_pct:.1f}%", delta_color="inverse")

# PV-spezifisch
st.write("### ☀️ PV-Optimierung")
col4, col5, col6, col7 = st.columns(4)
with col4:
    st.metric("PV-Überschuss gesamt", f"{pv_kpis['pv_surplus_total']:,.0f} kWh")
    st.metric("PV-Überschuss genutzt", f"{pv_kpis['pv_used_total']:,.0f} kWh")
with col5:
    st.metric("PV-Nutzungsgrad", f"{pv_kpis['pv_utilization_rate']:.1f}%")
    st.metric("PV-Anteil am Laden", f"{pv_kpis['pv_share_in_charging']:.1f}%")
with col6:
    st.metric("Abregelung vermieden", f"{pv_kpis['pv_used_total']:,.0f} kWh")
    st.metric("Abregelung verbleibend", f"{pv_kpis['pv_curtailed_total']:,.0f} kWh")
with col7:
    st.metric("PV-Nutzungswert (EEG)", f"{pv_kpis['pv_value']:,.0f} €")
    st.metric("Arbitrage-Erlös", f"{pv_kpis['arbitrage_revenue']:,.0f} €")

# Zyklen & Effizienz
st.write("### 🔄 Zyklen-Analyse")
col8, col9, col10, col11 = st.columns(4)
with col8:
    st.metric("PV-Zyklen", f"{pv_kpis['pv_cycles']:.2f}")
    st.metric("Grid-Arbitrage-Zyklen", f"{pv_kpis['grid_cycles']:.2f}")
with col9:
    st.metric("Gesamt-Arbitrage-Zyklen", f"{pv_kpis['total_cycles']:.2f}")
    st.metric("Baseline-Zyklen", f"{baseline_cycles:.2f}")
with col10:
    st.metric("Gesamt-Zyklen (real)", f"{total_cycles_real:.2f}")
    st.metric("Gesamt-Zyklen (frei)", f"{cycles_free:.2f}")

# Effizienzen
if pv_kpis['e_ch_pv_total'] > 0:
    efficiency_pv = pv_kpis['e_dis_total'] / pv_kpis['e_ch_pv_total'] * 100.0
else:
    efficiency_pv = 0.0
if pv_kpis['e_ch_grid_total'] > 0:
    efficiency_grid = pv_kpis['e_dis_total'] / pv_kpis['e_ch_grid_total'] * 100.0
else:
    efficiency_grid = 0.0
if (pv_kpis['e_ch_pv_total'] + pv_kpis['e_ch_grid_total']) > 0:
    efficiency_total = pv_kpis['e_dis_total'] / (pv_kpis['e_ch_pv_total'] + pv_kpis['e_ch_grid_total']) * 100.0
else:
    efficiency_total = 0.0
if energy_charged_free > 0:
    efficiency_free = energy_discharged_free / energy_charged_free * 100.0
else:
    efficiency_free = 0.0

with col11:
    st.metric("RTE (PV-Pfad)", f"{efficiency_pv:.1f}%")
    st.metric("RTE (gesamt)", f"{efficiency_total:.1f}%")

# EEG-Opportunitätskosten & Nettobeitrag
st.write("### 📑 EEG-Vergütung: Opportunitätskosten & Nettobeitrag")
ee1, ee2, ee3, ee4 = st.columns(4)
with ee1:
    st.metric("EEG-Verlust durch BESS", f"{eeg_loss_eur:,.0f} €")
    st.metric("EEG-Export ohne BESS (Baseline)", f"{baseline_eeg_export_eur:,.0f} €")
    st.metric("EEG-Verlust [% von EEG-Baseline]", f"{eeg_loss_pct_baseline:.1f}%")
with ee2:
    st.metric("EEG-Export mit BESS", f"{with_bess_eeg_export_eur:,.0f} €")
    st.metric("Netto-Erlös ggü. EEG-Baseline", f"{net_uplift_vs_eeg_eur:,.0f} €")
    st.metric("EEG-Export-Reduktion [%]", f"{eeg_export_reduction_pct:.1f}%")
with ee3:
    st.metric("Ø EEG-Wert genutzte PV", f"{eeg_value_weighted:.1f} €/MWh")
    st.metric("Break-even Entladepreis", f"{break_even_price_avg:.1f} €/MWh")
with ee4:
    pv_dis_share = (result["e_dis_pv_kwh"].sum() / pv_kpis["e_dis_total"] * 100.0) if pv_kpis["e_dis_total"] > 0 else 0.0
    st.metric("Ø Entladepreis (PV-Anteil)", f"{avg_p_dis_pv:.1f} €/MWh")
    st.metric("PV-Anteil an Entladung", f"{pv_dis_share:.1f}%")
    st.metric("Netto-Uplift ggü. EEG-Baseline [%]", f"{net_uplift_vs_eeg_pct:.1f}%")

# Weitere Analysen
st.write("### 🔍 Weitere Analysen")

avg_price_charge_grid = (
    (result.loc[result["e_ch_grid_kwh"] > 0, "price_eur_per_mwh"] * result.loc[result["e_ch_grid_kwh"] > 0, "e_ch_grid_kwh"]).sum()
    / pv_kpis['e_ch_grid_total'] if pv_kpis['e_ch_grid_total'] > 0 else 0.0 )

avg_price_charge_free = (
    (result_free.loc[result_free["e_ch_free_kwh"] > 0, "price_eur_per_mwh"] * result_free.loc[result_free["e_ch_free_kwh"] > 0, "e_ch_free_kwh"]).sum()
    / energy_charged_free if energy_charged_free > 0 else 0.0 )

avg_price_discharge = (
    (result.loc[result["e_dis_kwh"] > 0, "price_eur_per_mwh"] * result.loc[result["e_dis_kwh"] > 0, "e_dis_kwh"]).sum()
    / pv_kpis['e_dis_total'] if pv_kpis['e_dis_total'] > 0 else 0.0 )

avg_price_discharge_free = (
    (result_free.loc[result_free["e_dis_free_kwh"] > 0, "price_eur_per_mwh"] * result_free.loc[result_free["e_dis_free_kwh"] > 0, "e_dis_free_kwh"]).sum()
    / energy_discharged_free if energy_discharged_free > 0 else 0.0 )

max_net_load = float(result["net_total_kw"].abs().max())

price_spread = (avg_price_discharge - avg_price_charge_grid) if (avg_price_charge_grid > 0) else 0.0
price_spread_free = (avg_price_discharge_free - avg_price_charge_free) if (avg_price_charge_free > 0) else 0.0

# Kapazitätsnutzung
total_capacity = len(result) * P_max * dt_h
used_capacity_pv = result["p_ch_pv_kw"].sum() * dt_h
used_capacity_grid = (result["p_ch_grid_kw"].sum() + result["p_dis_kw"].sum()) * dt_h
used_capacity_total = used_capacity_pv + used_capacity_grid
used_capacity_free = (result_free["p_ch_free_kw"].sum() + result_free["p_dis_free_kw"].sum()) * dt_h

capacity_utilization_total = (used_capacity_total / total_capacity * 100.0) if total_capacity > 0 else 0.0
capacity_utilization_free = (used_capacity_free / total_capacity * 100.0) if total_capacity > 0 else 0.0

# =============================================================
# ---------------------- Visualisierung -----------------------
# =============================================================
st.subheader("📈 Zeitreihendarstellung (erste 7 Tage)")

sample_hours = min(7*24, len(result))
sample_slots = max(1, int(sample_hours / dt_h))
plot_data = result.iloc[:sample_slots].copy()

if "ts" in plot_data.columns:
    x_axis = "ts:T"
    plot_data = plot_data.reset_index()
else:
    plot_data = plot_data.reset_index()
    x_axis = "index:Q"

# Preise
price_chart = alt.Chart(plot_data).mark_line().encode(
    x=alt.X(x_axis, title="Zeit"),
    y=alt.Y("price_eur_per_mwh:Q", title="Preis [€/MWh]")
).properties(height=150, title="Day-Ahead Preise")

# PV-Erzeugung und -Nutzung (ohne pv_generation, wenn saldiert)
id_col = "index" if "ts" not in plot_data.columns else "ts"
x_axis_cat_or_time = f"{id_col}:O" if id_col == "index" else f"{id_col}:T"

pv_series = ["pv_surplus_available_kw", "pv_surplus_used_kw"]
if not pv_in_netload and "pv_generation_kw" in plot_data.columns:
    pv_series = ["pv_generation_kw"] + pv_series

pv_data = plot_data[[id_col] + pv_series].melt(
    id_vars=[id_col],
    value_vars=pv_series,
    var_name="Typ", value_name="Leistung"
)

pv_chart = alt.Chart(pv_data).mark_line().encode(
    x=alt.X(x_axis_cat_or_time, title="Zeit"),
    y=alt.Y("Leistung:Q", title="PV/Überschuss [kW]"),
    color=alt.Color("Typ:N", legend=alt.Legend(title="PV/Überschuss"))
).properties(height=200, title="PV-Überschuss (hergeleitet) und -Nutzung")

# BESS-Leistung
power_data = plot_data[[id_col, "p_ch_pv_kw", "p_ch_grid_kw", "p_dis_kw"]].melt(
    id_vars=[id_col],
    value_vars=["p_ch_pv_kw", "p_ch_grid_kw", "p_dis_kw"],
    var_name="Typ", value_name="Leistung"
)

power_chart = alt.Chart(power_data).mark_bar().encode(
    x=alt.X(x_axis_cat_or_time, title="Zeit"),
    y=alt.Y("Leistung:Q", title="BESS-Leistung [kW]"),
    color=alt.Color("Typ:N", legend=alt.Legend(title="BESS-Betrieb"))
).properties(height=200, title="BESS Lade-/Entladeleistung (PV-First)")

# SoC
soc_chart = alt.Chart(plot_data).mark_line().encode(
    x=alt.X(x_axis, title="Zeit"),
    y=alt.Y("soc_total_kwh:Q", title="SoC [kWh]", scale=alt.Scale(domain=[0, E_max]))
).properties(height=150, title="Batterie-SoC (gesamt)")

# Netzlast
net_chart = alt.Chart(plot_data).mark_line().encode(
    x=alt.X(x_axis, title="Zeit"),
    y=alt.Y("net_total_kw:Q", title="Netzlast [kW]")
).properties(height=150, title="Gesamte Netzlast (ohne PV-Laden)")

# Charts anzeigen
st.altair_chart(price_chart, use_container_width=True)
st.altair_chart(pv_chart, use_container_width=True)
st.altair_chart(power_chart, use_container_width=True)
st.altair_chart(soc_chart, use_container_width=True)
st.altair_chart(net_chart, use_container_width=True)

# =============================================================
# --------------------------- Export --------------------------
# =============================================================
st.subheader("💾 Export")

# KPI-Tabelle für Export
kpi_rows = []
kpi_rows.append(("Erlös PV-First [€]", round(pv_kpis['total_revenue'], 2)))
kpi_rows.append(("Erlös freier Handel [€]", round(revenue_free, 2)))
kpi_rows.append(("Verlust absolut [€]", round(revenue_gap_abs, 2)))
kpi_rows.append(("Verlust relativ [%]", round(revenue_gap_pct, 1)))

kpi_rows.append(("PV-Überschuss gesamt [kWh]", round(pv_kpis['pv_surplus_total'], 1)))
kpi_rows.append(("PV-Überschuss genutzt [kWh]", round(pv_kpis['pv_used_total'], 1)))
kpi_rows.append(("PV-Nutzungsgrad [%]", round(pv_kpis['pv_utilization_rate'], 1)))
kpi_rows.append(("PV-Anteil am Laden [%]", round(pv_kpis['pv_share_in_charging'], 1)))
kpi_rows.append(("Abregelung vermieden [kWh]", round(pv_kpis['pv_used_total'], 1)))
kpi_rows.append(("Abregelung verbleibend [kWh]", round(pv_kpis['pv_curtailed_total'], 1)))

kpi_rows.append(("Energie geladen (PV) [kWh]", round(pv_kpis['e_ch_pv_total'], 1)))
kpi_rows.append(("Energie geladen (Grid) [kWh]", round(pv_kpis['e_ch_grid_total'], 1)))
kpi_rows.append(("Energie geladen (frei) [kWh]", round(energy_charged_free, 1)))

kpi_rows.append(("Energie entladen [kWh]", round(pv_kpis['e_dis_total'], 1)))
kpi_rows.append(("Energie entladen (frei) [kWh]", round(energy_discharged_free, 1)))

kpi_rows.append(("PV-Zyklen [#]", round(pv_kpis['pv_cycles'], 2)))
kpi_rows.append(("Grid-Arbitrage-Zyklen [#]", round(pv_kpis['grid_cycles'], 2)))
kpi_rows.append(("Gesamt-Arbitrage-Zyklen [#]", round(pv_kpis['total_cycles'], 2)))

kpi_rows.append(("Baseline-Zyklen [#]", round(baseline_cycles, 2)))
kpi_rows.append(("Gesamt-Zyklen real [#]", round(total_cycles_real, 2)))
kpi_rows.append(("Gesamt-Zyklen frei [#]", round(cycles_free, 2)))

kpi_rows.append(("RTE gesamt [%]", round(efficiency_total, 1)))
kpi_rows.append(("RTE frei [%]", round(efficiency_free, 1)))

kpi_rows.append(("Arbitrage-Erlös [€]", round(pv_kpis['arbitrage_revenue'], 2)))
kpi_rows.append(("PV-Nutzungswert (EEG) [€]", round(pv_kpis['pv_value'], 2)))
kpi_rows.append(("Gesamt-Nutzen [€]", round(pv_kpis['total_revenue'], 2)))

kpi_rows.append(("Ø Grid-Lade-Preis [€/MWh]", round(avg_price_charge_grid, 1)))
kpi_rows.append(("Ø Lade-Preis frei [€/MWh]", round(avg_price_charge_free, 1)))

kpi_rows.append(("Ø Entlade-Preis [€/MWh]", round(avg_price_discharge, 1)))
kpi_rows.append(("Ø Entlade-Preis frei [€/MWh]", round(avg_price_discharge_free, 1)))

kpi_rows.append(("Grid-Preis-Spread [€/MWh]", round(price_spread, 1)))
kpi_rows.append(("Preis-Spread frei [€/MWh]", round(price_spread_free, 1)))

kpi_rows.append(("Max. Netzlast [kW]", round(max_net_load, 1)))
kpi_rows.append(("Kapazitätsnutzung real [%]", round(capacity_utilization_total, 1)))
kpi_rows.append(("Kapazitätsnutzung frei [%]", round(capacity_utilization_free, 1)))

kpi_rows.append(("RTE [%]", rte_pct))
kpi_rows.append(("E_max [kWh]", E_max))
kpi_rows.append(("P_max [kW]", P_max))
kpi_rows.append(("P_conn [kW]", P_conn))
kpi_rows.append(("Zyklenlimit [#/Jahr]", cycles_cap))
kpi_rows.append(("EEG aktiv [ja/nein]", "ja" if enable_eeg else "nein"))
kpi_rows.append(("EEG-Vergütungssatz [€/MWh]", eeg_tariff))
kpi_rows.append(("Ø EEG-Einspeisewert für genutzte PV [€/MWh]", round(eeg_value_weighted, 1)))
kpi_rows.append(("Stunden mit negativen Preisen [#]", neg_price_hours))

# EEG-spezifische KPI-Zusätze
kpi_rows.append(("EEG-Verlust durch BESS [€]", round(eeg_loss_eur, 2)))
kpi_rows.append(("EEG-Export (ohne BESS) [€]", round(baseline_eeg_export_eur, 2)))
kpi_rows.append(("EEG-Export (mit BESS) [€]", round(with_bess_eeg_export_eur, 2)))
kpi_rows.append(("Netto-Erlös ggü. EEG-Baseline [€]", round(net_uplift_vs_eeg_eur, 2)))
kpi_rows.append(("Ø Entladepreis (PV-Anteil) [€/MWh]", round(avg_p_dis_pv, 1)))
kpi_rows.append(("Break-even Entladepreis [€/MWh]", round(break_even_price_avg, 1)))

# Prozentwerte in den Excel-Export
kpi_rows.append(("EEG-Verlust [% von EEG-Baseline]", round(eeg_loss_pct_baseline, 1)))
kpi_rows.append(("EEG-Export-Reduktion [%]", round(eeg_export_reduction_pct, 1)))
kpi_rows.append(("Netto-Uplift ggü. EEG-Baseline [%]", round(net_uplift_vs_eeg_pct, 1)))

kpi_summary = pd.DataFrame(kpi_rows, columns=["Kennzahl","Wert"])

# Excel-Export erstellen
bio = BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as writer:
    result.to_excel(writer, index=False, sheet_name="Zeitreihen_PV_First")
    result_free.to_excel(writer, index=False, sheet_name="Zeitreihen_FreierHandel")
    kpi_summary.to_excel(writer, index=False, sheet_name="KPIs_Vergleich")

bio.seek(0)

st.download_button(
    "📥 Ergebnisse als Excel herunterladen",
    data=bio,
    file_name="BESS_PV_First_Arbitrage_Ergebnisse.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
