# qca_results.py
"""
UI de Resultados QCA: Soluciones C/I/P (Compleja, Intermedia, Parsimoniosa).
TFG — Analítica Cualitativa Comparada
"""

import streamlit as st
import pandas as pd
import altair as alt

from core.qca_engine import QCAEngine, term_to_string

# ============================================================
# Helper visual
# ============================================================

def metric_card(title, value, subtitle=None):
    st.metric(
        label=f"**{title}**",
        value=f"{value:.3f}" if value is not None else "—",
        help=subtitle
    )

# ============================================================
# UI principal
# ============================================================

def show():

    st.title("QCA — Resultados de Minimización")

    # Validaciones de datos previos
    if "truth_table_filtered" not in st.session_state:
        st.warning("Generá la Truth Table primero.")
        return
    if "conditions" not in st.session_state:
        st.warning("No se encontraron condiciones. Volvé a Inicio.")
        return

    truth_table = st.session_state["truth_table_filtered"]
    conditions = st.session_state["conditions"]

    if truth_table.empty:
        st.error("No hay configuraciones después de los filtros de la truth table.")
        return

    st.success("Truth Table cargada correctamente. Módulo QCA listo para ejecutar minimización.")

    # ============================================================
    # Construir motor QCA
    # ============================================================

    st.markdown("### Motor de Minimización")
    engine = QCAEngine(truth_table, conditions)

    # Direccional expectations (opcional)
    st.markdown("#### Directional Expectations (opcional)")

    with st.expander("Establecer supuestos teóricos (I)"):
        de = {}
        for cond in conditions:
            val = st.selectbox(
                f"{cond}:",
                options=["No especificar", "Presencia favorece (+)", "Ausencia favorece (–)"],
                index=0
            )
            if val == "Presencia favorece (+)":
                de[cond] = 1
            elif val == "Ausencia favorece (–)":
                de[cond] = 0
            else:
                de[cond] = None

    # ============================================================
    # Ejecutar soluciones
    # ============================================================

    st.markdown("---")
    st.header("📌 Soluciones C / I / P")

    btn_run = st.button("🔍 Ejecutar Minimización")

    if not btn_run:
        st.info("Presioná el botón para generar las soluciones.")
        return

    with st.spinner("Calculando soluciones…"):

        sol_C = engine.complex_solution()
        sol_P = engine.parsimonious_solution()
        sol_I = engine.intermediate_solution(directional_expectations=de)

    st.success("Soluciones generadas correctamente.")

    # ============================================================
    # Display – Cards de Métricas
    # ============================================================

    def show_solution_block(label, solution):
        st.markdown(f"## {label} Solution — **({solution['type']})**")
        st.markdown(f"### Expresión booleana")
        st.code(solution["expression"])

        # Métricas
        m = solution["metrics"]
        colA, colB, colC = st.columns(3)
        with colA:
            metric_card("Consistency", m["Consistency"], "Grado en que la solución es suficiente")
        with colB:
            metric_card("Coverage", m["Coverage"], "Grado en que explica el outcome")
        with colC:
            metric_card("PRI", m["PRI"], "Reducción de inconsistencias")

        # Expansor técnico
        with st.expander("Ver términos implicantes"):
            df_terms = pd.DataFrame([
                {"Term": term_to_string(t), **t} for t in solution["terms"]
            ])
            st.dataframe(df_terms)

        st.markdown("---")

    # Mostrar cada solución
    show_solution_block("Complex (C)", sol_C)
    show_solution_block("Intermediate (I)", sol_I)
    show_solution_block("Parsimonious (P)", sol_P)

    # ============================================================
    # Visualización global
    # ============================================================

    st.header("📈 Visualización Comparativa")

    df_plot = pd.DataFrame([
        {"Solution": "Complex (C)", **sol_C["metrics"]},
        {"Solution": "Intermediate (I)", **sol_I["metrics"]},
        {"Solution": "Parsimonious (P)", **sol_P["metrics"]},
    ])

    chart = alt.Chart(df_plot).mark_bar().encode(
        x="Solution:N",
        y="Consistency:Q",
        tooltip=["Consistency","Coverage","PRI"],
        color="Solution:N"
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    # ============================================================
    # Exportación
    # ============================================================

    st.header("⬇️ Exportar Soluciones")

    export_dict = {
        "Complex": sol_C,
        "Intermediate": sol_I,
        "Parsimonious": sol_P
    }

    df_export = pd.DataFrame({
        "Solution": ["C", "I", "P"],
        "Expression": [sol_C["expression"], sol_I["expression"], sol_P["expression"]],
        "Consistency": [sol_C["metrics"]["Consistency"], sol_I["metrics"]["Consistency"], sol_P["metrics"]["Consistency"]],
        "Coverage": [sol_C["metrics"]["Coverage"], sol_I["metrics"]["Coverage"], sol_P["metrics"]["Coverage"]],
        "PRI": [sol_C["metrics"]["PRI"], sol_I["metrics"]["PRI"], sol_P["metrics"]["PRI"]],
    })

    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", csv, "qca_solutions.csv", "text/csv")

    st.success("Listo. Podés avanzar con Robustez o Reportes.")
