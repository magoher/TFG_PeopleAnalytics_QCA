# robustness_check.py
"""
Robustness & Sensitivity Analysis para QCA
TFG — Analítica Cualitativa Comparada

Incluye:
    1. Robustez por variación de umbrales (Consistency, PRI, Frecuencia)
    2. Robustez por eliminación de casos
    3. Robustez por randomización del outcome
    4. Métrica de estabilidad estructural entre soluciones
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from core.qca_engine import QCAEngine, term_to_string

# ============================================================
# Helpers
# ============================================================

def jaccard_terms(solA, solB):
    """
    Mide similitud entre dos soluciones basadas en términos implicantes.
    """
    setA = set([term_to_string(t) for t in solA])
    setB = set([term_to_string(t) for t in solB])
    if len(setA) + len(setB) == 0:
        return 0
    return len(setA & setB) / len(setA | setB)

# ============================================================
# UI principal
# ============================================================

def show():

    st.title("Robustness & Sensitivity Analysis (QCA)")

    if "truth_table" not in st.session_state:
        st.warning("Primero generá la Truth Table.")
        return

    if "conditions" not in st.session_state:
        st.warning("Regresá al inicio para seleccionar condiciones.")
        return

    truth_table_original = st.session_state["truth_table"].copy()
    conditions = st.session_state["conditions"]

    st.markdown("""
    Este módulo evalúa la **solidez epistemológica** del modelo QCA:
    - variación de umbrales
    - eliminación de casos
    - permutación del outcome
    - estabilidad de la expresión causal
    """)

    # -------------------------------------------------------------------
    # SECTION 1: Varying Thresholds
    # -------------------------------------------------------------------

    st.header("Robustez por Variación de Umbrales")

    with st.expander("Opciones de Sensibilidad"):
        cons_range = st.slider("Rango de Consistencia", 0.5, 0.95, (0.75, 0.85), 0.01)
        pri_range = st.slider("Rango de PRI", 0.5, 0.95, (0.65, 0.80), 0.01)
        freq_min = st.number_input("Frecuencia mínima", 1, 10, 1)

    # Preparar grilla
    cons_vals = np.linspace(cons_range[0], cons_range[1], 5)
    pri_vals = np.linspace(pri_range[0], pri_range[1], 5)

    sensitivity_records = []

    if st.button("▶ Ejecutar Sensibilidad por Umbrales"):

        with st.spinner("Calculando estabilidad…"):

            for cmin in cons_vals:
                for pmin in pri_vals:

                    # Filtrar truth table según umbrales
                    tt = truth_table_original.copy()
                    tt_filtered = tt[
                        (tt["Consistency_raw"] >= cmin) &
                        (tt["PRI"] >= pmin) &
                        (tt["N_cases"] >= freq_min)
                    ]

                    if len(tt_filtered) == 0:
                        continue

                    engine = QCAEngine(tt_filtered, conditions)
                    sol_C = engine.complex_solution()
                    sol_P = engine.parsimonious_solution()

                    stability = jaccard_terms(sol_C["terms"], sol_P["terms"])

                    sensitivity_records.append({
                        "Consistency_thr": cmin,
                        "PRI_thr": pmin,
                        "Stability_CvsP": stability
                    })

        df_sens = pd.DataFrame(sensitivity_records)
        st.session_state["df_sens"] = df_sens

        st.success("Análisis completado.")

        if not df_sens.empty:
            chart = alt.Chart(df_sens).mark_rect().encode(
                x="Consistency_thr:O",
                y="PRI_thr:O",
                color=alt.Color("Stability_CvsP:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Consistency_thr", "PRI_thr", "Stability_CvsP"]
            )
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(df_sens)

    # -------------------------------------------------------------------
    # SECTION 2: Case Deletion Robustness
    # -------------------------------------------------------------------

    st.header("Robustez por Eliminación de Casos")

    if st.button("▶ Ejecutar análisis de eliminación"):

        with st.spinner("Analizando estabilidad con eliminación de casos…"):

            df = truth_table_original.copy()

            baseline_engine = QCAEngine(df, conditions)
            baseline_solution = baseline_engine.complex_solution()

            deletion_records = []

            for idx in df.index:
                df_dropped = df.drop(idx).reset_index(drop=True)
                engine = QCAEngine(df_dropped, conditions)
                sol = engine.complex_solution()

                stability = jaccard_terms(baseline_solution["terms"], sol["terms"])

                deletion_records.append({
                    "Removed_case": idx,
                    "Stability": stability
                })

            df_del = pd.DataFrame(deletion_records)

        st.success("Eliminación de casos completada.")

        chart2 = alt.Chart(df_del).mark_line(point=True).encode(
            x="Removed_case:O",
            y="Stability:Q",
            tooltip=["Removed_case","Stability"]
        )
        st.altair_chart(chart2, use_container_width=True)

        st.dataframe(df_del)

    # -------------------------------------------------------------------
    # SECTION 3: Randomization Test
    # -------------------------------------------------------------------

    st.header("Randomization Test (Aleatorización del Outcome)")

    reps = st.slider("Número de permutaciones", 10, 200, 50)

    if st.button("▶ Ejecutar Randomization Test"):

        with st.spinner("Ejecutando permutaciones…"):

            df_rand_results = []

            df = truth_table_original.copy()
            engine_baseline = QCAEngine(df, conditions)
            base_sol = engine_baseline.complex_solution()

            Y = df["OUT_mean"].copy()

            for r in range(reps):
                df_perm = df.copy()
                df_perm["OUT_mean"] = np.random.permutation(Y)

                engine = QCAEngine(df_perm, conditions)
                sol = engine.complex_solution()

                stability = jaccard_terms(base_sol["terms"], sol["terms"])

                df_rand_results.append({
                    "Run": r,
                    "Stability": stability
                })

        df_rand = pd.DataFrame(df_rand_results)

        st.success("Randomization test completado.")

        chart3 = alt.Chart(df_rand).mark_bar().encode(
            x="Run:O",
            y="Stability:Q",
            tooltip=["Run","Stability"]
        )
        st.altair_chart(chart3, use_container_width=True)

        st.dataframe(df_rand)

        st.markdown(f"""
            **Estabilidad media:** {df_rand['Stability'].mean():.3f}
        """)

    # -------------------------------------------------------------------
    # SECTION 4: Export
    # -------------------------------------------------------------------

    st.markdown("---")
    st.header("Exportar Resultados de Robustez")

    if "df_sens" in st.session_state:
        csv = st.session_state["df_sens"].to_csv(index=False).encode("utf-8")
        st.download_button("Descargar Sensibilidad (CSV)", csv, "robustness_thresholds.csv")

    st.info("Podés continuar con el módulo de Reportes Automáticos.")
