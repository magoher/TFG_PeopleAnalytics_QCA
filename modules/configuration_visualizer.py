# configuration_visualizer.py
"""
Configuration Visualizer for QCA Platform
Advanced visualizer for configurations and implicant terms
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import itertools

# Internal dependencies
from core.qca_engine import QCAEngine, term_to_string

# ----------------------------
# Helper functions
# ----------------------------

def parse_boolean_term(term_str):
    """
    Parse a boolean textual term like "A*~B*C" into dict {'A':1, 'B':0, 'C':1}
    """
    if isinstance(term_str, dict):
        return term_str
    
    term_str = str(term_str).strip()
    if term_str == "":
        return {}
    
    parts = [p.strip() for p in term_str.split("*") if p.strip() != ""]
    d = {}
    for p in parts:
        if p.startswith("~"):
            d[p[1:]] = 0
        else:
            d[p] = 1
    return d


def build_term_dataframe(terms, truth_table, conditions):
    """
    Build DataFrame with metrics for each implicant term.
    """
    rows = []
    df = truth_table.copy()
    
    # Ensure Y is numpy array
    Y = np.array(df["OUT_mean"].values, dtype=float)

    for t in terms:
        # Parse term
        term = t if isinstance(t, dict) else parse_boolean_term(t)
        
        # Build membership for each row
        membership = np.ones(len(df), dtype=float)
        
        for cond in conditions:
            if cond in term:
                v = term[cond]
                cond_values = np.array(df[f"{cond}_avg"].values, dtype=float)
                
                if v == 1:
                    # Use np.multiply for element-wise multiplication
                    membership = np.multiply(membership, cond_values)
                elif v == 0:
                    # Use 1 - cond_values
                    membership = np.multiply(membership, 1.0 - cond_values)
            # If condition not in term, it's a "don't care" - multiply by 1.0 (no change)
        
        # Calculate metrics
        # Convert to numpy arrays for safe operations
        membership_arr = np.array(membership, dtype=float)
        Y_arr = np.array(Y, dtype=float)
        
        # Use np.minimum safely
        min_values = np.minimum(membership_arr, Y_arr)
        numer = float(np.sum(min_values))
        denom_term = float(np.sum(membership_arr))
        denom_Y = float(np.sum(Y_arr))
        
        consistency = numer / denom_term if denom_term > 0 else float("nan")
        coverage = numer / denom_Y if denom_Y > 0 else float("nan")
        
        # Count cases with membership > threshold
        n_cases_real = int(np.sum(membership_arr > 0.0001))

        row = {
            "Term_txt": term_to_string(term) if isinstance(term, dict) else str(t),
            "N_cases": n_cases_real,
            "Consistency": consistency,
            "Coverage": coverage,
        }

        # Average membership per condition
        for cond in conditions:
            cond_values = np.array(df[f"{cond}_avg"].values, dtype=float)
            if np.sum(membership_arr) > 0:
                avg_val = np.sum(cond_values * membership_arr) / np.sum(membership_arr)
            else:
                avg_val = 0.0
            row[f"{cond}_avg"] = float(avg_val)

        rows.append(row)

    return pd.DataFrame(rows)


def build_node_link_structure(terms, conditions):
    """
    Build nodes and links for a simple diagram.
    """
    nodes = []
    links = []
    
    # Condition nodes
    for i, cond in enumerate(conditions):
        nodes.append({
            "id": f"C:{cond}", 
            "label": cond, 
            "type": "condition", 
            "group": 0, 
            "index": i
        })

    # Term nodes
    for j, term in enumerate(terms):
        term_txt = term_to_string(term) if isinstance(term, dict) else str(term)
        nodes.append({
            "id": f"T:{j}", 
            "label": term_txt, 
            "type": "term", 
            "group": 1, 
            "index": j
        })

    # Outcome node
    nodes.append({
        "id": "O:OUTCOME", 
        "label": "Outcome", 
        "type": "outcome", 
        "group": 2, 
        "index": 0
    })

    # Condition -> Term links
    for j, term in enumerate(terms):
        term_dict = term if isinstance(term, dict) else parse_boolean_term(term)
        for cond in conditions:
            if cond in term_dict:
                links.append({
                    "source": f"C:{cond}", 
                    "target": f"T:{j}", 
                    "weight": 1, 
                    "label": f"{cond}={term_dict[cond]}"
                })

    # Term -> Outcome links
    for j, term in enumerate(terms):
        links.append({
            "source": f"T:{j}", 
            "target": "O:OUTCOME", 
            "weight": 1, 
            "label": "suffices"
        })

    return pd.DataFrame(nodes), pd.DataFrame(links)


def simplify_expression(expr):
    """Clean expression string for display."""
    return str(expr).replace("  ", " ").strip()


# ----------------------------
# UI & visual blocks
# ----------------------------

def show():
    st.title("🗺️ Configuration Visualizer")
    
    # Basic checks
    if "truth_table_filtered" not in st.session_state:
        st.warning("Generate the truth table in 'Truth Table' first.")
        return
    
    if "conditions" not in st.session_state:
        st.warning("No conditions found. Go to 'Start' first.")
        return
    
    truth_table = st.session_state["truth_table_filtered"].copy()
    conditions = st.session_state["conditions"]
    
    if truth_table.empty:
        st.error("No configurations after filtering. Adjust filters in 'Truth Table'.")
        return
    
    st.markdown("Use this module to **visually explore** configurations, terms, and causal routes produced by QCA.")
    
    # Compute QCA solutions if not in state
    if "qca_solutions" in st.session_state:
        sols = st.session_state["qca_solutions"]
        sol_C = sols.get("Complex")
        sol_I = sols.get("Intermediate")
        sol_P = sols.get("Parsimonious")
    else:
        try:
            engine = QCAEngine(truth_table, conditions)
            sol_C = engine.complex_solution()
            sol_P = engine.parsimonious_solution()
            sol_I = engine.intermediate_solution()
            
            # Store solutions
            st.session_state["qca_solutions"] = {
                "Complex": sol_C,
                "Intermediate": sol_I,
                "Parsimonious": sol_P
            }
        except Exception as e:
            st.error(f"Error creating QCA engine: {e}")
            return
    
    # Sidebar controls
    st.sidebar.header("Visualizer Controls")
    chosen_solution = st.sidebar.selectbox(
        "Select solution type", 
        ["Complex", "Intermediate", "Parsimonious"], 
        index=0
    )
    
    show_heatmap = st.sidebar.checkbox("Show truth table heatmap", value=True)
    show_config_map = st.sidebar.checkbox("Show configuration map", value=True)
    show_term_panel = st.sidebar.checkbox("Show implicant terms panel", value=True)
    show_causal_diagram = st.sidebar.checkbox("Show causal diagram", value=True)
    show_venn = st.sidebar.checkbox("Show Venn diagram (≤3 conditions)", value=False)
    
    # Map chosen solution
    solution_map = {
        "Complex": st.session_state["qca_solutions"]["Complex"],
        "Intermediate": st.session_state["qca_solutions"]["Intermediate"],
        "Parsimonious": st.session_state["qca_solutions"]["Parsimonious"]
    }
    
    selected_solution = solution_map.get(chosen_solution)
    
    if selected_solution is None:
        st.warning("Selected solution not available.")
        return
    
    # Display solution info
    st.subheader(f"Selected Solution: {chosen_solution}")
    st.markdown("**Expression:**")
    st.code(simplify_expression(selected_solution.get("expression", "")))
    
    metrics = selected_solution.get("metrics", {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        consistency = metrics.get("Consistency", float("nan"))
        st.metric("Consistency", f"{consistency:.3f}" if not np.isnan(consistency) else "—")
    
    with col2:
        coverage = metrics.get("Coverage", float("nan"))
        st.metric("Coverage", f"{coverage:.3f}" if not np.isnan(coverage) else "—")
    
    with col3:
        pri = metrics.get("PRI", float("nan"))
        st.metric("PRI", f"{pri:.3f}" if not np.isnan(pri) else "—")
    
    st.markdown("---")
    
    # -------------------------
    # Visual block 1: Truth table heatmap
    # -------------------------
    if show_heatmap:
        st.markdown("### 🔥 Heatmap: Average membership per condition (by configuration)")
        
        # Prepare data
        heat_df = truth_table[["CONFIG", "N_cases", "Consistency_raw", "Coverage_raw"] + 
                             [f"{c}_avg" for c in conditions]].copy()
        
        # Melt for heatmap
        heat_melt = heat_df.melt(
            id_vars=["CONFIG", "N_cases", "Consistency_raw", "Coverage_raw"],
            var_name="Condition", 
            value_name="Membership"
        )
        
        # Clean condition names
        heat_melt["Condition"] = heat_melt["Condition"].str.replace("_avg", "", regex=False)
        
        # Create heatmap
        heat_chart = alt.Chart(heat_melt).mark_rect().encode(
            x=alt.X("Condition:N", sort=conditions, title="Condition"),
            y=alt.Y("CONFIG:N", 
                   sort=alt.EncodingSortField(field="Consistency_raw", op="max", order="descending"),
                   title="Configuration"),
            color=alt.Color("Membership:Q", scale=alt.Scale(scheme="blues"), title="Membership"),
            tooltip=[
                "CONFIG", 
                "Condition", 
                alt.Tooltip("Membership:Q", format=".3f"), 
                "N_cases", 
                alt.Tooltip("Consistency_raw:Q", format=".3f")
            ]
        ).properties(
            height=40 * min(30, heat_df.shape[0]), 
            width=900
        )
        
        st.altair_chart(heat_chart, use_container_width=True)
        st.markdown("**Quick interpretation:** darker colors → higher average condition membership in the configuration.")
    
    # -------------------------
    # Visual block 2: Configuration map
    # -------------------------
    if show_config_map:
        st.markdown("### 🗺️ Configuration Map — Consistency vs Coverage")
        
        cfg = truth_table.copy().reset_index(drop=True)
        
        if cfg.empty:
            st.info("No configurations to display.")
        else:
            # Create scatter plot
            cfg_plot = alt.Chart(cfg).mark_circle(size=120).encode(
                x=alt.X("Consistency_raw:Q", scale=alt.Scale(domain=[0,1]), title="Consistency (raw)"),
                y=alt.Y("Coverage_raw:Q", scale=alt.Scale(domain=[0,1]), title="Coverage (raw)"),
                size=alt.Size("N_cases:Q", title="# cases", scale=alt.Scale(range=[50,1000])),
                color=alt.Color("Contradictory:N", legend=alt.Legend(title="Contradictory")),
                tooltip=[
                    "CONFIG", 
                    "N_cases", 
                    alt.Tooltip("Consistency_raw:Q", format=".3f"), 
                    alt.Tooltip("Coverage_raw:Q", format=".3f"), 
                    "Contradictory"
                ]
            ).properties(width=700, height=450)
            
            # Add labels
            labels = cfg_plot.mark_text(align='left', dx=7, dy=-7).encode(text="CONFIG:N")
            st.altair_chart(cfg_plot + labels, use_container_width=True)
            
            # Configuration selector
            st.markdown("Select a configuration to explore:")
            selected_cfg = st.selectbox("Choose configuration (CONFIG)", options=cfg["CONFIG"].tolist())
            
            if selected_cfg:
                row = cfg[cfg["CONFIG"] == selected_cfg].iloc[0]
                st.write(f"**Selected:** {selected_cfg} — N_cases: {int(row['N_cases'])}, "
                        f"Consistency: {row['Consistency_raw']:.3f}, Coverage: {row['Coverage_raw']:.3f}")
                
                # Show per-condition membership
                mems = {c: float(row[f"{c}_avg"]) for c in conditions}
                df_mems = pd.DataFrame({
                    "Condition": list(mems.keys()), 
                    "Membership": list(mems.values())
                })
                
                bar = alt.Chart(df_mems).mark_bar().encode(
                    x="Condition:N", 
                    y=alt.Y("Membership:Q", scale=alt.Scale(domain=[0,1])), 
                    tooltip=["Condition", "Membership"]
                )
                st.altair_chart(bar, use_container_width=True)
    
    # -------------------------
    # Visual block 3: Implicant terms
    # -------------------------
    if show_term_panel:
        st.markdown("### 🧾 Implicant Terms Panel")
        
        terms = selected_solution.get("terms", [])
        if terms:
            term_df = build_term_dataframe(terms, truth_table, conditions)
            term_df_sorted = term_df.sort_values("Consistency", ascending=False).reset_index(drop=True)
            
            # Display terms
            display_cols = ["Term_txt", "N_cases", "Consistency", "Coverage"] + [f"{c}_avg" for c in conditions]
            st.dataframe(term_df_sorted[display_cols].round(3))
            
            # Term detail selector
            st.markdown("Select a term for details:")
            term_choice = st.selectbox("Term", options=term_df_sorted["Term_txt"].tolist(), index=0)
            
            if term_choice:
                detail_row = term_df_sorted[term_df_sorted["Term_txt"] == term_choice].iloc[0]
                
                st.markdown(f"**Term:** {term_choice}")
                st.write(f"N cases: {int(detail_row['N_cases'])}")
                st.write(f"Consistency (term): {detail_row['Consistency']:.3f}")
                st.write(f"Coverage (term): {detail_row['Coverage']:.3f}")
                
                # Membership by condition bar chart
                mems = {c: float(detail_row[f"{c}_avg"]) for c in conditions}
                df_mems = pd.DataFrame({
                    "Condition": list(mems.keys()), 
                    "Avg_Membership": list(mems.values())
                })
                
                bar = alt.Chart(df_mems).mark_bar().encode(
                    x=alt.X("Condition:N", sort=conditions),
                    y=alt.Y("Avg_Membership:Q", scale=alt.Scale(domain=[0,1])),
                    tooltip=["Condition", alt.Tooltip("Avg_Membership:Q", format=".3f")]
                ).properties(height=300)
                
                st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No terms in this solution.")
    
    # -------------------------
    # Visual block 4: Causal diagram
    # -------------------------
    if show_causal_diagram:
        st.markdown("### 🌳 Causal Diagram (simplified)")
        
        terms_for_diagram = selected_solution.get("terms", [])
        
        if len(terms_for_diagram) == 0:
            st.info("No terms to visualize.")
        else:
            nodes_df, links_df = build_node_link_structure(terms_for_diagram, conditions)
            
            # Simplified display (tables for now)
            st.write("**Nodes:**")
            st.dataframe(nodes_df[["id", "label", "type"]])
            
            st.write("**Links:**")
            st.dataframe(links_df)
            
            st.info("Interactive diagram visualization is in development. Export the data for external visualization tools.")
    
    # -------------------------
    # Visual block 5: Venn diagram (simplified)
    # -------------------------
    if show_venn:
        st.markdown("### ⚖️ Venn Diagram (for 2-3 conditions)")
        
        if len(conditions) <= 1:
            st.info("Need at least 2 conditions for Venn diagram.")
        elif len(conditions) > 3:
            st.info("Venn diagrams for more than 3 conditions are not supported in this module.")
        else:
            if "calibrated_df" not in st.session_state:
                st.warning("Need 'calibrated_df' for Venn diagram construction.")
            else:
                cal = st.session_state["calibrated_df"].copy()
                
                # Create boolean matrix
                bool_df = pd.DataFrame({c: (cal[c] >= 0.5).astype(int) for c in conditions})
                
                # Count intersections
                combos = []
                for bits in itertools.product([0, 1], repeat=len(conditions)):
                    mask = (bool_df == pd.Series(bits, index=bool_df.columns)).all(axis=1)
                    count = int(mask.sum())
                    combos.append({
                        "combo": "".join(map(str, bits)),
                        "bits": bits,
                        "count": count
                    })
                
                combos_df = pd.DataFrame(combos)
                st.write("Intersection counts:")
                st.dataframe(combos_df)
                
                st.info("Export these values and create the diagram in your preferred visualization tool.")
    
    # -------------------------
    # Automatic interpretation
    # -------------------------
    st.markdown("---")
    st.header("📝 Automatic Interpretation")
    
    def auto_interpret(solution):
        expr = solution.get("expression", "")
        metrics = solution.get("metrics", {})
        terms_local = solution.get("terms", [])
        n_terms = len(terms_local)
        
        interpretation = []
        interpretation.append(f"The selected solution contains **{n_terms}** term(s) with Consistency = {metrics.get('Consistency', float('nan')):.3f}, Coverage = {metrics.get('Coverage', float('nan')):.3f}.")
        
        if n_terms == 0:
            interpretation.append("No implicant terms identified in this solution.")
            return "\n\n".join(interpretation)
        
        interpretation.append("Term summary:")
        term_df_local = build_term_dataframe(terms_local, truth_table, conditions)
        term_df_local = term_df_local.sort_values("Consistency", ascending=False).reset_index(drop=True)
        
        for i, row in term_df_local.head(5).iterrows():
            interpretation.append(f"- **{row['Term_txt']}**: covers ~{int(row['N_cases'])} configurations with Consistency {row['Consistency']:.3f} and Coverage {row['Coverage']:.3f}.")
        
        interpretation.append("\n**General interpretation:**")
        if len(term_df_local) > 0:
            top_term = term_df_local.iloc[0]
            interpretation.append(f"The strongest path is **{top_term['Term_txt']}**, suggesting that the combination {top_term['Term_txt'].replace('*', ' and ')} is consistently associated with the outcome.")
        
        if len(term_df_local) > 1:
            interpretation.append("There are alternative, less consistent paths that also contribute to explaining the outcome; examine robustness and stability of these paths.")
        
        interpretation.append("**Recommended verification:** Check sensitivity to threshold changes and case removal (Robustness module).")
        
        return "\n\n".join(interpretation)
    
    interpretation_text = auto_interpret(selected_solution)
    st.markdown(interpretation_text)
    
    # -------------------------
    # Export panel
    # -------------------------
    st.markdown("---")
    st.header("⬇️ Export Results")
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        # Export truth table
        csv_tt = truth_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Truth Table (CSV)", 
            csv_tt, 
            "truth_table_filtered.csv", 
            "text/csv"
        )
    
    with colB:
        # Export terms
        if selected_solution.get("terms"):
            term_rows_export = build_term_dataframe(selected_solution.get("terms", []), truth_table, conditions)
            csv_terms = term_rows_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Terms (CSV)", 
                csv_terms, 
                "qca_terms.csv", 
                "text/csv"
            )
        else:
            st.info("No terms to export")
    
    with colC:
        # Export node-link data
        if selected_solution.get("terms"):
            nds, lks = build_node_link_structure(selected_solution.get("terms", []), conditions)
            export_json = {
                "nodes": nds.to_dict(orient="records"), 
                "links": lks.to_dict(orient="records")
            }
            st.download_button(
                "Download Node-Link (JSON)", 
                json.dumps(export_json, indent=2, ensure_ascii=False).encode("utf-8"), 
                "node_link.json", 
                "application/json"
            )
        else:
            st.info("No node-link data")
    
    st.success("Visualization complete. Review interpretations, export results, and continue with Robustness or Reports.")


# Direct execution
if __name__ == "__main__":
    show()