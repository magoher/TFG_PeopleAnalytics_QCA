import streamlit as st
import pandas as pd

# ============================================================
#  START PAGE — CONTROL PANEL OF ANALYSIS
# ============================================================

def show():

    st.title("Data Input & Setup")
    st.write("Carga tus datos y define el tipo de análisis que deseas realizar.")

    # ============================================================
    # 1. UPLOAD DATA
    # ============================================================

    st.header("1. Cargar archivo")
    uploaded_file = st.file_uploader("Subí tu archivo CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Esperando archivo CSV…")
        return

    df = pd.read_csv(uploaded_file)
    st.success(f"Archivo cargado: **{uploaded_file.name}** ({df.shape[0]} filas, {df.shape[1]} columnas)")

    # Guardamos dataset temporalmente
    st.session_state["raw_df"] = df

    with st.expander("Ver primeras filas"):
        st.dataframe(df.head())

    # ============================================================
    # 2. SELECT ANALYSIS TYPE
    # ============================================================

    st.header("2. Seleccionar tipo de análisis")

    analysis_type = st.selectbox(
        "Elegí el análisis principal:",
        [
            "QCA — Análisis Cualitativo Comparado",
            "NCA — Necessary Condition Analysis",
            "Calibración Avanzada",
            "Tabla de Verdad",
            "Minimización (Solución C, I, P)",
            "Robustez y Sensibilidad",
            "Reporte Automático"
        ]
    )

    st.session_state["analysis_type"] = analysis_type

    # ============================================================
    # 3. TARGET / OUTCOME SELECTION
    # ============================================================

    st.header("3. Seleccionar condición resultado (Outcome)")

    columns = df.columns.tolist()

    outcome_col = st.selectbox(
        "Elegí la columna resultado (Y):",
        columns,
        index=0
    )

    st.session_state["outcome"] = outcome_col

    # ============================================================
    # 4. CONDITIONS SELECTION
    # ============================================================

    st.header("4. Seleccionar condiciones (X)")

    conditions = st.multiselect(
        "Seleccioná las condiciones explicativas:",
        [col for col in columns if col != outcome_col],
        default=[col for col in columns if col != outcome_col]
    )

    if len(conditions) == 0:
        st.warning("Debe seleccionar al menos una condición.")
        return

    st.session_state["conditions"] = conditions

    # ============================================================
    # 5. CONTINUE BUTTON
    # ============================================================

    st.markdown("---")
    if st.button("➡ Continuar", type="primary"):
        st.session_state["ready_for_analysis"] = True
        st.success("¡Perfecto! Ahora podés ir al siguiente módulo desde el menú lateral.")

        st.balloons()


# For direct execution
if __name__ == "__main__":
    show()
