# truth_table.py
"""
Truth Table Generator para QCA — TFG People Analytics.

REQUISITOS:
    st.session_state['calibrated_df']    -> dataframe con condiciones calibradas y outcome membership
    st.session_state['conditions']       -> lista de condiciones
    st.session_state['outcome']          -> outcome original
    st.session_state['_OUTCOME_MEMBERSHIP'] -> generado automáticamente en calibration.py

EXPORTA:
    st.session_state['truth_table']      -> DataFrame de truth table completa
    st.session_state['truth_table_filtered'] -> DataFrame tras filtros

FUNCIONES:
    - Construye Tabla de Verdad Crisp o Fuzzy-like (binarización con threshold 0.5)
    - Calcula métricas: Consistency, PRI, Coverage, Raw N
    - Detecta contradicciones
    - Visualiza la estructura de configuraciones
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import StringIO
import json

# -------------------------------------------------------------------
# Utilidades para métricas de QCA
# -------------------------------------------------------------------

def consistency_raw(cond_membership, outcome_membership):
    """Raw consistency for sufficiency: sum(min(Xi, Yi)) / sum(Xi)."""
    X = np.array(cond_membership, dtype=float)
    Y = np.array(outcome_membership, dtype=float)
    denom = X.sum()
    if denom == 0:
        return np.nan
    numer = np.minimum(X, Y).sum()
    return float(numer / denom)

def consistency_pri(cond_membership, outcome_membership):
    """PRI consistency: reduces cases where X is subset of ~Y."""
    X = np.array(cond_membership, dtype=float)
    Y = np.array(outcome_membership, dtype=float)
    numer = np.minimum(X, Y).sum()
    denom = np.minimum(X, 1 - Y).sum() + numer
    if denom == 0:
        return np.nan
    return float(numer / denom)

def coverage_raw(cond_membership, outcome_membership):
    """Raw coverage: sum(min(Xi, Yi)) / sum(Yi)."""
    X = np.array(cond_membership, dtype=float)
    Y = np.array(outcome_membership, dtype=float)
    denom = Y.sum()
    if denom == 0:
        return np.nan
    numer = np.minimum(X, Y).sum()
    return float(numer / denom)

# -------------------------------------------------------------------
# Construcción de configuraciones (binarias o fuzzy-binarizadas)
# -------------------------------------------------------------------

def binarize_fuzzy_for_truth_table(series, threshold=0.5):
    """
    Binariza membresías fuzzy para truth table: 1 si >= threshold.
    """
    return (series >= threshold).astype(int)

# -------------------------------------------------------------------
# UI principal
# -------------------------------------------------------------------

def show():
    st.title("📊 Truth Table — Tabla de Verdad para QCA")

    # Validación mínima
    if "calibrated_df" not in st.session_state:
        st.warning("Primero calibrá tus condiciones en 'Calibración'.")
        return
    if "conditions" not in st.session_state:
        st.warning("No se encontraron condiciones. Regresá a Inicio.")
        return

    calibrated_df = st.session_state["calibrated_df"].copy()
    conditions = st.session_state["conditions"]

    if "_OUTCOME_MEMBERSHIP" not in calibrated_df.columns:
        st.error("El outcome no está calibrado o no existe '_OUTCOME_MEMBERSHIP'.")
        return

    st.markdown("Esta sección genera la **truth table** basada en las condiciones calibradas.")

    # Preview
    with st.expander("📄 Ver DataFrame calibrado"):
        st.dataframe(calibrated_df)

    outcome_mem = calibrated_df["_OUTCOME_MEMBERSHIP"]

    st.markdown("---")
    st.subheader("⚙️ Opciones de generación")

    # Modo crisp o fuzzy-binarizado
    mode = st.radio("Modo de truth table", ["Binarización fuzzy (>=0.5)", "Usar crisp puro"], index=0)
    binary_threshold = 0.5

    if mode == "Usar crisp puro":
        st.info("Si tus condiciones no eran crisp originalmente, se usarán los valores calibrados (0/1).")
        binary_threshold = 0.5  # crisp ya viene como 0/1

    # -------------------------------------------------------------------
    # Generar configuraciones binarias
    # -------------------------------------------------------------------

    st.subheader("🧮 Construyendo configuraciones...")

    bin_df = pd.DataFrame(index=calibrated_df.index)

    for cond in conditions:
        series = calibrated_df[cond].astype(float)
        bin_df[cond] = binarize_fuzzy_for_truth_table(series, threshold=binary_threshold)

    # Configuración como string
    bin_df["CONFIG"] = bin_df[conditions].astype(str).agg("".join, axis=1)

    # Outcome crispificado para truth table
    bin_df["OUT"] = binarize_fuzzy_for_truth_table(outcome_mem, threshold=binary_threshold)

    # -------------------------------------------------------------------
    # Agrupación
    # -------------------------------------------------------------------

    grouped = bin_df.groupby("CONFIG")
    rows = []

    for config, group in grouped:
        cond_memberships = []
        for cond in conditions:
            # For each configuration row, original membership (fuzzy) is needed
            cond_memberships.append(calibrated_df.loc[group.index, cond].values)

        # membership de la conjunción fuzzy (min en paralelo)
        conj = np.min(cond_memberships, axis=0)

        row = {
            "CONFIG": config,
            "N_cases": len(group),
            "Consistency_raw": consistency_raw(conj, outcome_mem.loc[group.index]),
            "PRI": consistency_pri(conj, outcome_mem.loc[group.index]),
            "Coverage_raw": coverage_raw(conj, outcome_mem.loc[group.index]),
            "OUT_mean": float(outcome_mem.loc[group.index].mean()),
        }

        # contradicción si en esa config hay outcomes crisp 0 y 1 mezclados
        outcomes_crisp = bin_df.loc[group.index, "OUT"].unique().tolist()
        row["Contradictory"] = 1 if (len(outcomes_crisp) > 1) else 0

        # agregar membership promedio por condición
        for cond in conditions:
            row[f"{cond}_avg"] = float(calibrated_df.loc[group.index, cond].mean())

        rows.append(row)

    truth_table = pd.DataFrame(rows).sort_values("CONFIG").reset_index(drop=True)

    # Guardar sin filtrar
    st.session_state["truth_table"] = truth_table.copy()

    st.success("Truth table generada correctamente.")

    st.markdown("---")
    st.subheader("🎛️ Filtros")

    colA, colB, colC = st.columns(3)
    with colA:
        min_cons = st.slider("Consistencia mínima", 0.0, 1.0, 0.75, 0.01)
    with colB:
        min_pri = st.slider("PRI mínima", 0.0, 1.0, 0.65, 0.01)
    with colC:
        min_freq = st.number_input("Frecuencia mínima", min_value=1, value=1)

    exclude_contradictions = st.checkbox("Excluir configuraciones contradictorias", value=True)

    # Aplicar filtros
    filt = (
        (truth_table["Consistency_raw"] >= min_cons) &
        (truth_table["PRI"] >= min_pri) &
        (truth_table["N_cases"] >= min_freq)
    )

    if exclude_contradictions:
        filt = filt & (truth_table["Contradictory"] == 0)

    truth_filtered = truth_table[filt].reset_index(drop=True)
    st.session_state["truth_table_filtered"] = truth_filtered

    st.markdown("### Resultados filtrados")
    st.dataframe(truth_filtered)

    st.markdown("---")
    st.subheader("📈 Visualización")

    # Chart: Consistency vs Coverage
    chart = alt.Chart(truth_filtered).mark_circle(size=120).encode(
        x=alt.X("Consistency_raw:Q", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("Coverage_raw:Q", scale=alt.Scale(domain=[0,1])),
        color="Contradictory:N",
        tooltip=["CONFIG","N_cases","Consistency_raw","Coverage_raw","PRI"]
    )
    st.altair_chart(chart, use_container_width=True)

    # Matrix-like config visualization
    st.markdown("### Mapa de configuraciones")

    if len(truth_filtered) > 0:
        melt = truth_filtered[["CONFIG"] + [c+"_avg" for c in conditions]].copy()
        melt = melt.melt(id_vars="CONFIG", var_name="Condition", value_name="Membership")

        map_chart = alt.Chart(melt).mark_rect().encode(
            x="Condition:N",
            y="CONFIG:N",
            color=alt.Color("Membership:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["CONFIG","Condition","Membership"]
        )
        st.altair_chart(map_chart, use_container_width=True)
    else:
        st.info("No hay configuraciones después de los filtros.")

    st.markdown("---")
    st.subheader("⬇️ Exportar")

    col1, col2 = st.columns(2)
    with col1:
        csv = truth_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", csv, "truth_table_filtered.csv", "text/csv")

    with col2:
        json_str = truth_filtered.to_json(orient="records", indent=2)
        st.download_button("Descargar JSON", json_str, "truth_table_filtered.json", "application/json")

    st.info("Una vez estés conforme, avanzá a 'Análisis QCA' para la minimización booleana.")


# end of file
