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
    st.title("Truth Table — Tabla de Verdad para QCA")

    # ============================================================
    # VALIDACIÓN ROBUSTA DE DATOS REQUERIDOS
    # ============================================================
    
    # Lista de datos requeridos y su módulo de origen
    required_data = [
        ("raw_df", "Start", "Upload dataset first"),
        ("conditions", "Start", "Select conditions in Start module"),
        ("outcome", "Start", "Select outcome variable in Start module"),
        ("calibrated_df", "Calibration", "Apply calibration in Calibration module first")
    ]
    
    # Verificar cada dato requerido
    missing_data = []
    for data_key, module_name, message in required_data:
        if data_key not in st.session_state:
            missing_data.append((data_key, module_name, message))
        elif st.session_state[data_key] is None:
            missing_data.append((data_key, module_name, f"Data is None - {message}"))
    
    # Si faltan datos, mostrar mensaje claro
    if missing_data:
        st.error("Missing or incomplete data. Please complete these steps first:")
        
        for data_key, module_name, message in missing_data:
            st.markdown(f"**{data_key}** → From '{module_name}' module: {message}")
        
        # Botones de navegación sugeridos
        st.markdown("### Recommended Actions:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Go to Start Page", use_container_width=True):
                st.session_state.navigation = "Start"
                st.rerun()
        
        with col2:
            if "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
                if st.button("Go to Calibration", use_container_width=True):
                    st.session_state.navigation = "Calibration"
                    st.rerun()
        
        return
    
    # ============================================================
    # VERIFICACIÓN ADICIONAL DE ESTRUCTURA DE DATOS
    # ============================================================
    
    calibrated_df = st.session_state["calibrated_df"].copy()
    conditions = st.session_state["conditions"]
    
    # Verificar que calibrated_df tiene las columnas esperadas
    missing_columns = []
    for cond in conditions:
        if cond not in calibrated_df.columns:
            missing_columns.append(cond)
    
    if missing_columns:
        st.error(f"The following conditions are missing from calibrated data: {', '.join(missing_columns)}")
        st.info("Please re-calibrate your data in the Calibration module.")
        return
    
    if "_OUTCOME_MEMBERSHIP" not in calibrated_df.columns:
        st.error("Outcome membership column '_OUTCOME_MEMBERSHIP' not found in calibrated data.")
        st.info("The outcome variable may not have been calibrated properly.")
        return
    
    # ============================================================
    # RESUMEN DE DATOS DISPONIBLES
    # ============================================================
    
    st.markdown("### Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cases", calibrated_df.shape[0])
    with col2:
        st.metric("Conditions", len(conditions))
    with col3:
        st.metric("Configuration Space", 2**len(conditions))
    with col4:
        outcome_present = calibrated_df["_OUTCOME_MEMBERSHIP"].notna().sum()
        st.metric("Valid Outcomes", outcome_present)
    
    st.markdown("This section generates the **truth table** based on calibrated conditions.")

    # Preview
    with st.expander("View Calibrated Data"):
        st.dataframe(calibrated_df.head(15))

    outcome_mem = calibrated_df["_OUTCOME_MEMBERSHIP"]

    st.markdown("---")
    st.subheader("Generation Options")

    # Modo crisp o fuzzy-binarizado
    mode = st.radio("Truth table mode", ["Fuzzy binarization (>=0.5)", "Pure crisp"], index=0)
    binary_threshold = 0.5

    if mode == "Pure crisp":
        st.info("If conditions were not originally crisp, calibrated values (0/1) will be used.")
        binary_threshold = 0.5  # crisp ya viene como 0/1

    # -------------------------------------------------------------------
    # Generar configuraciones binarias
    # -------------------------------------------------------------------

    st.subheader("Building configurations...")

    with st.spinner("Processing configurations..."):
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

    st.subheader("Calculating metrics...")
    
    with st.spinner("Calculating consistency and coverage..."):
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

    st.success(f"Truth table generated successfully. Found {len(truth_table)} unique configurations.")

    st.markdown("---")
    st.subheader("Configuration Filtering")

    colA, colB, colC = st.columns(3)
    with colA:
        min_cons = st.slider("Minimum consistency", 0.0, 1.0, 0.75, 0.01)
    with colB:
        min_pri = st.slider("Minimum PRI", 0.0, 1.0, 0.65, 0.01)
    with colC:
        min_freq = st.number_input("Minimum frequency", min_value=1, value=1)

    exclude_contradictions = st.checkbox("Exclude contradictory configurations", value=True)

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

    st.markdown(f"### Filtered Results ({len(truth_filtered)} configurations)")
    st.dataframe(truth_filtered, use_container_width=True)

    st.markdown("---")
    st.subheader("Visualization")

    if len(truth_filtered) > 0:
        # Chart: Consistency vs Coverage
        chart = alt.Chart(truth_filtered).mark_circle(size=120).encode(
            x=alt.X("Consistency_raw:Q", scale=alt.Scale(domain=[0,1]), title="Consistency"),
            y=alt.Y("Coverage_raw:Q", scale=alt.Scale(domain=[0,1]), title="Coverage"),
            color="Contradictory:N",
            tooltip=["CONFIG","N_cases","Consistency_raw","Coverage_raw","PRI"]
        ).properties(title="Consistency vs Coverage")
        st.altair_chart(chart, use_container_width=True)

        # Matrix-like config visualization
        st.markdown("### Configuration Map")

        melt = truth_filtered[["CONFIG"] + [c+"_avg" for c in conditions]].copy()
        melt = melt.melt(id_vars="CONFIG", var_name="Condition", value_name="Membership")

        map_chart = alt.Chart(melt).mark_rect().encode(
            x="Condition:N",
            y="CONFIG:N",
            color=alt.Color("Membership:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["CONFIG","Condition","Membership"]
        ).properties(title="Average Membership per Configuration")
        st.altair_chart(map_chart, use_container_width=True)
    else:
        st.info("No configurations remain after applying filters.")

    st.markdown("---")
    st.subheader("Export Options")

    col1, col2 = st.columns(2)
    with col1:
        csv = truth_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "truth_table_filtered.csv", "text/csv")

    with col2:
        json_str = truth_filtered.to_json(orient="records", indent=2)
        st.download_button("Download JSON", json_str, "truth_table_filtered.json", "application/json")

    st.info("Once satisfied with the truth table, proceed to 'QCA Solutions' for Boolean minimization.")


# end of file