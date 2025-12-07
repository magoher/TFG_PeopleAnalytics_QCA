"""
QCA Results UI: C/I/P Solutions (Complex, Intermediate, Parsimonious)
TFG — Qualitative Comparative Analysis Platform
"""

import streamlit as st
import pandas as pd
import altair as alt

from core.qca_engine import QCAEngine, term_to_string

# ============================================================
# Helper functions
# ============================================================

def metric_card(title, value, subtitle=None):
    """Display a metric card with consistent formatting."""
    st.metric(
        label=f"**{title}**",
        value=f"{value:.3f}" if value is not None else "—",
        help=subtitle
    )

def validate_prerequisites():
    """Validate required data is available in session state."""
    required_keys = [
        ("truth_table_filtered", "Generate Truth Table first in the 'Truth Table' module"),
        ("conditions", "Select conditions in the 'Start' module"),
        ("outcome", "Select outcome variable in the 'Start' module")
    ]
    
    missing = []
    for key, message in required_keys:
        if key not in st.session_state:
            missing.append((key, message))
        elif st.session_state[key] is None:
            missing.append((key, f"{key} is None - {message}"))
    
    return missing

# ============================================================
# Main UI
# ============================================================

def show():
    st.title("QCA — Minimization Results")

    # ============================================================
    # Data Validation
    # ============================================================
    missing_data = validate_prerequisites()
    
    if missing_data:
        st.error("Missing or incomplete data. Please complete these steps first:")
        
        for key, message in missing_data:
            st.markdown(f"**{key}** → {message}")
        
        # Navigation suggestions
        st.markdown("### Recommended Actions:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Go to Start", use_container_width=True):
                st.session_state.navigation = "Start"
                st.rerun()
        
        with col2:
            if "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
                if st.button("Go to Truth Table", use_container_width=True):
                    st.session_state.navigation = "Truth Table"
                    st.rerun()
        
        return
    
    truth_table = st.session_state["truth_table_filtered"]
    conditions = st.session_state["conditions"]
    
    if truth_table.empty:
        st.error("No configurations remain after truth table filtering.")
        st.info("Adjust your consistency and frequency thresholds in the Truth Table module.")
        return
    
    # ============================================================
    # Data Summary
    # ============================================================
    st.success(f"Truth Table loaded successfully. Ready for QCA minimization.")
    
    with st.expander("Data Summary"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Configurations", len(truth_table))
        with col2:
            st.metric("Conditions", len(conditions))
        with col3:
            total_cases = truth_table["N_cases"].sum()
            st.metric("Total Cases", total_cases)
    
    # ============================================================
    # Initialize QCA Engine
    # ============================================================
    st.markdown("### QCA Minimization Engine")
    engine = QCAEngine(truth_table, conditions)
    
    # ============================================================
    # Directional Expectations (Optional)
    # ============================================================
    st.markdown("#### Directional Expectations (Optional)")
    
    with st.expander("Set theoretical assumptions for Intermediate Solution"):
        de = {}
        for cond in conditions:
            val = st.selectbox(
                f"{cond}:",
                options=["Not specified", "Presence contributes to outcome (+)", "Absence contributes to outcome (–)"],
                index=0,
                key=f"de_{cond}"
            )
            if val == "Presence contributes to outcome (+)":
                de[cond] = 1
            elif val == "Absence contributes to outcome (–)":
                de[cond] = 0
            else:
                de[cond] = None
        
        if any(v is not None for v in de.values()):
            st.info("Directional expectations will be used for Intermediate solution only.")
    
    # ============================================================
    # Generate Solutions
    # ============================================================
    st.markdown("---")
    st.header("C / I / P Solutions")
    
    col_run, col_info = st.columns([1, 3])
    with col_run:
        btn_run = st.button("Run Minimization", type="primary", use_container_width=True)
    
    with col_info:
        st.caption("Complex (C): All logical remainders excluded")
        st.caption("Intermediate (I): Uses directional expectations (if specified)")
        st.caption("Parsimonious (P): All logical remainders included")
    
    if not btn_run:
        st.info("Click the button above to generate QCA solutions.")
        return
    
    with st.spinner("Calculating solutions... This may take a moment for complex configurations."):
        sol_C = engine.complex_solution()
        sol_P = engine.parsimonious_solution()
        sol_I = engine.intermediate_solution(directional_expectations=de)
    
    # ============================================================
    # Save Solutions to Session State
    # ============================================================
    qca_solutions = {
        "Complex": sol_C,
        "Intermediate": sol_I,
        "Parsimonious": sol_P,
        "directional_expectations": de,
        "conditions": conditions
    }
    st.session_state["qca_solutions"] = qca_solutions
    
    st.success("QCA solutions generated successfully.")
    
    # ============================================================
    # Display Solutions - Individual Blocks
    # ============================================================
    def show_solution_block(label, solution):
        """Display a solution block with metrics and expression."""
        st.markdown(f"## {label} Solution — **({solution['type']})**")
        
        # Boolean Expression
        st.markdown("#### Boolean Expression")
        st.code(solution["expression"], language="text")
        
        # Metrics
        m = solution["metrics"]
        colA, colB, colC = st.columns(3)
        with colA:
            metric_card("Consistency", m["Consistency"], "Degree to which solution is sufficient")
        with colB:
            metric_card("Coverage", m["Coverage"], "Degree to which solution explains outcome")
        with colC:
            metric_card("PRI", m["PRI"], "Reduction of inconsistencies")
        
        # Prime Implicants
        with st.expander("View Prime Implicants"):
            if solution["terms"]:
                df_terms = pd.DataFrame([
                    {"Term": term_to_string(t), **t} for t in solution["terms"]
                ])
                st.dataframe(df_terms, use_container_width=True)
            else:
                st.info("No prime implicants found for this solution.")
        
        st.markdown("---")
    
    # Display each solution
    show_solution_block("Complex (C)", sol_C)
    show_solution_block("Intermediate (I)", sol_I)
    show_solution_block("Parsimonious (P)", sol_P)
    
    # ============================================================
    # Comparative Visualization
    # ============================================================
    st.header("Comparative Visualization")
    
    # Prepare data for plotting
    df_plot = pd.DataFrame([
        {"Solution": "Complex (C)", **sol_C["metrics"]},
        {"Solution": "Intermediate (I)", **sol_I["metrics"]},
        {"Solution": "Parsimonious (P)", **sol_P["metrics"]},
    ])
    
    # Bar chart comparing consistency
    chart = alt.Chart(df_plot).mark_bar(size=50).encode(
        x=alt.X("Solution:N", title="Solution Type", sort=["Complex (C)", "Intermediate (I)", "Parsimonious (P)"]),
        y=alt.Y("Consistency:Q", title="Consistency", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Solution:N", legend=None),
        tooltip=["Solution", "Consistency", "Coverage", "PRI"]
    ).properties(
        height=350,
        title="Solution Consistency Comparison"
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # ============================================================
    # Export Solutions
    # ============================================================
    st.header("Export Solutions")
    
    # Create export DataFrame
    df_export = pd.DataFrame({
        "Solution": ["C", "I", "P"],
        "Type": ["Complex", "Intermediate", "Parsimonious"],
        "Expression": [sol_C["expression"], sol_I["expression"], sol_P["expression"]],
        "Consistency": [sol_C["metrics"]["Consistency"], sol_I["metrics"]["Consistency"], sol_P["metrics"]["Consistency"]],
        "Coverage": [sol_C["metrics"]["Coverage"], sol_I["metrics"]["Coverage"], sol_P["metrics"]["Coverage"]],
        "PRI": [sol_C["metrics"]["PRI"], sol_I["metrics"]["PRI"], sol_P["metrics"]["PRI"]],
        "Number_of_Terms": [len(sol_C["terms"]), len(sol_I["terms"]), len(sol_P["terms"])]
    })
    
    # Display export options
    col_csv, col_json, col_summary = st.columns(3)
    
    with col_csv:
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "qca_solutions.csv",
            "text/csv",
            help="Download solutions as CSV"
        )
    
    with col_json:
        export_json = {
            "solutions": {
                "Complex": sol_C,
                "Intermediate": sol_I,
                "Parsimonious": sol_P
            },
            "metadata": {
                "conditions": conditions,
                "timestamp": pd.Timestamp.now().isoformat(),
                "configurations_count": len(truth_table)
            }
        }
        import json
        json_str = json.dumps(export_json, indent=2, default=str)
        st.download_button(
            "Download JSON",
            json_str,
            "qca_solutions.json",
            "application/json",
            help="Download complete solution data as JSON"
        )
    
    with col_summary:
        st.info(f"""
        **Summary:**
        - **Configurations:** {len(truth_table)}
        - **Conditions:** {len(conditions)}
        - **Best Consistency:** {max(sol_C['metrics']['Consistency'], sol_I['metrics']['Consistency'], sol_P['metrics']['Consistency']):.3f}
        """)
    
    # Next steps suggestion
    st.markdown("---")
    st.markdown("### Next Steps")
    
    if "qca_solutions" in st.session_state:
        st.success("Solutions saved. You can now proceed to:")
        
        col_viz, col_robust, col_report = st.columns(3)
        
        with col_viz:
            if st.button("Configuration Visualizer", use_container_width=True):
                st.session_state.navigation = "Visualizer"
                st.rerun()
        
        with col_robust:
            if st.button("Robustness Analysis", use_container_width=True):
                st.session_state.navigation = "Robustness"
                st.rerun()
        
        with col_report:
            if st.button("Generate Report", use_container_width=True):
                st.session_state.navigation = "Reports"
                st.rerun()


# For direct execution
if __name__ == "__main__":
    show()