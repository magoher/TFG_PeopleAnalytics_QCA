# calibration.py (versión completa corregida)
"""
Calibration Module for QCA Platform
Transforms variables to fuzzy/crisp membership for set-theoretic analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import altair as alt
from io import StringIO


# ============================================
# CALIBRATION FUNCTIONS
# ============================================

def calibrate_crisp(series: pd.Series, threshold: float, higher_is_in=True):
    """Crisp calibration: 1 if condition met, 0 otherwise."""
    if higher_is_in:
        return (series >= threshold).astype(float)
    else:
        return (series <= threshold).astype(float)


def calibrate_fuzzy_linear(series: pd.Series, full_non, crossover, full_mem):
    """Fuzzy linear calibration (piecewise linear) in range [0,1]."""
    s = series.astype(float).copy()
    memb = np.zeros(len(s), dtype=float)

    # Avoid division by zero
    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    # Regions
    left_mask = s <= full_non
    right_mask = s >= full_mem
    mid_low_mask = (s > full_non) & (s <= crossover)
    mid_high_mask = (s > crossover) & (s < full_mem)

    memb[left_mask] = 0.0
    memb[right_mask] = 1.0

    # Linear from full_non -> crossover (0 -> 0.5)
    if (crossover - full_non) > 0:
        memb[mid_low_mask] = 0.5 * safe_div((s[mid_low_mask] - full_non), (crossover - full_non))

    # Linear from crossover -> full_mem (0.5 -> 1)
    if (full_mem - crossover) > 0:
        memb[mid_high_mask] = 0.5 + 0.5 * safe_div((s[mid_high_mask] - crossover), (full_mem - crossover))

    return np.clip(memb, 0.0, 1.0)


def suggest_percentiles(series: pd.Series, p_low=25, p_mid=50, p_high=75):
    """Suggest thresholds based on percentiles."""
    q_low = np.nanpercentile(series.dropna(), p_low)
    q_mid = np.nanpercentile(series.dropna(), p_mid)
    q_high = np.nanpercentile(series.dropna(), p_high)
    return float(q_low), float(q_mid), float(q_high)


def consistency_single_condition(condition_membership, outcome_membership):
    """Consistency metric for a single condition."""
    cond = np.array(condition_membership, dtype=float)
    out = np.array(outcome_membership, dtype=float)
    denom = cond.sum()
    if denom == 0:
        return np.nan
    numer = np.minimum(cond, out).sum()
    return float(numer / denom)


# ============================================
# MAIN INTERFACE
# ============================================

def show():
    st.title("🔧 Condition Calibration")
    
    # Validate previous data
    required_keys = ["raw_df", "conditions", "outcome"]
    missing = [k for k in required_keys if k not in st.session_state]
    
    if missing:
        st.error(f"Missing data: {', '.join(missing)}. Go to 'Start' first.")
        return
    
    df = st.session_state["raw_df"].copy()
    conditions = st.session_state["conditions"]
    outcome = st.session_state["outcome"]
    
    # Summary
    st.subheader("📋 Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Conditions", len(conditions))
    
    # Quick data view
    with st.expander("📄 View Data"):
        st.dataframe(df.head())
    
    st.divider()
    
    # ============================================
    # OUTCOME CALIBRATION (CORREGIDO)
    # ============================================
    st.subheader("🎯 Outcome Calibration")
    
    outcome_series = df[outcome]
    is_numeric = pd.api.types.is_numeric_dtype(outcome_series)
    
    # Initialize outcome function
    outcome_to_membership = None
    
    if is_numeric:
        # Fuzzy for numeric outcome
        out_low, out_mid, out_high = suggest_percentiles(outcome_series)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            out_full_non = st.number_input("Full non-membership", value=float(out_low), key="out_fn")
        with col_b:
            out_crossover = st.number_input("Crossover (0.5)", value=float(out_mid), key="out_cx")
        with col_c:
            out_full_mem = st.number_input("Full membership", value=float(out_high), key="out_fm")
        
        # Define numeric outcome function
        def numeric_outcome_func(series):
            return calibrate_fuzzy_linear(series, out_full_non, out_crossover, out_full_mem)
        
        outcome_to_membership = numeric_outcome_func
        
    else:
        # Crisp for categorical outcome
        cats = outcome_series.dropna().unique().tolist()
        chosen_cat = st.selectbox("Category for 'high' outcome:", cats, key="out_cat")
        
        # Define categorical outcome function
        def categorical_outcome_func(series):
            return (series == chosen_cat).astype(float).values
        
        outcome_to_membership = categorical_outcome_func
    
    # Verify outcome function was created
    if outcome_to_membership is None:
        st.error("Failed to create outcome calibration function")
        return
    
    # ============================================
    # CONDITION CALIBRATION
    # ============================================
    st.divider()
    st.subheader("⚙️ Condition-by-Condition Calibration")
    
    # General options
    with st.sidebar:
        st.subheader("⚡ Options")
        default_method = st.radio("Default method:", ["Fuzzy", "Crisp"])
        show_suggestions = st.checkbox("Show suggestions", value=True)
    
    # Dictionaries to store results
    calibration_config = {}
    calibrated_columns = {}
    
    for cond in conditions:
        st.markdown(f"#### {cond}")
        
        series = df[cond]
        is_num = pd.api.types.is_numeric_dtype(series)
        
        # Show statistics
        if is_num:
            stats = series.describe()
            st.caption(f"Min: {stats['min']:.2f} | Med: {stats['50%']:.2f} | Max: {stats['max']:.2f}")
        
        # Method selection
        method = st.radio(
            f"Method for {cond}",
            ["Fuzzy", "Crisp"],
            index=0 if default_method == "Fuzzy" else 1,
            key=f"method_{cond}",
            horizontal=True
        )
        
        if method == "Crisp":
            # Crisp Calibration
            if is_num:
                if show_suggestions:
                    suggested = np.nanpercentile(series.dropna(), 50)
                else:
                    suggested = series.median()
                
                threshold = st.number_input(
                    f"Threshold for {cond}",
                    value=float(suggested),
                    key=f"thresh_{cond}"
                )
                
                direction = st.radio(
                    f"Direction for {cond}",
                    ["Higher = 1", "Lower = 1"],
                    key=f"dir_{cond}",
                    horizontal=True
                )
                
                memb = calibrate_crisp(
                    series, 
                    threshold, 
                    higher_is_in=(direction == "Higher = 1")
                )
                
                calibration_config[cond] = {
                    "method": "crisp",
                    "threshold": float(threshold),
                    "higher_is_in": (direction == "Higher = 1")
                }
            else:
                # Categorical variable
                uniques = series.dropna().unique().tolist()
                selected = st.multiselect(
                    f"Values that are '1' for {cond}",
                    uniques,
                    default=uniques[:1] if uniques else [],
                    key=f"cat_{cond}"
                )
                
                memb = series.isin(selected).astype(float).values
                calibration_config[cond] = {
                    "method": "crisp_categorical",
                    "values": selected
                }
        
        else:  # Fuzzy Method
            if is_num:
                if show_suggestions:
                    low, mid, high = suggest_percentiles(series)
                else:
                    low, mid, high = series.min(), series.median(), series.max()
                
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    full_non = st.number_input(
                        f"Full non for {cond}",
                        value=float(low),
                        key=f"fn_{cond}"
                    )
                with col_f2:
                    crossover = st.number_input(
                        f"Crossover for {cond}",
                        value=float(mid),
                        key=f"cx_{cond}"
                    )
                with col_f3:
                    full_mem = st.number_input(
                        f"Full membership for {cond}",
                        value=float(high),
                        key=f"fm_{cond}"
                    )
                
                if not (full_non < crossover < full_mem):
                    st.error("Must satisfy: full_non < crossover < full_mem")
                    memb = np.full(len(series), 0.5)
                else:
                    memb = calibrate_fuzzy_linear(series, full_non, crossover, full_mem)
                
                calibration_config[cond] = {
                    "method": "fuzzy",
                    "full_non": float(full_non),
                    "crossover": float(crossover),
                    "full_mem": float(full_mem)
                }
            else:
                # Fuzzy for categorical (actually crisp)
                uniques = series.dropna().unique().tolist()
                selected = st.multiselect(
                    f"High values for {cond}",
                    uniques,
                    default=uniques[:1] if uniques else [],
                    key=f"fuzzy_cat_{cond}"
                )
                
                memb = series.isin(selected).astype(float).values
                calibration_config[cond] = {
                    "method": "fuzzy_categorical",
                    "values": selected
                }
        
        # Save calibrated column
        calibrated_columns[cond] = memb
        
        # Show preview and metric
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.caption("Calibration preview:")
            preview_df = pd.DataFrame({
                "Original": series.head(5).values,
                "Calibrated": memb[:5]
            })
            st.dataframe(preview_df, use_container_width=True)
        
        with col_right:
            # Calculate consistency
            try:
                outcome_mem = outcome_to_membership(df[outcome])
                cons = consistency_single_condition(memb, outcome_mem)
                if not np.isnan(cons):
                    st.metric("Consistency", f"{cons:.3f}")
                else:
                    st.caption("Consistency: N/A")
            except Exception as e:
                st.caption(f"Consistency not calculable: {str(e)}")
        
        st.divider()
    
    # ============================================
    # FINAL ACTIONS
    # ============================================
    st.subheader("💾 Save Calibration")
    
    col_save, col_export, col_reset = st.columns(3)
    
    with col_save:
        if st.button("💾 Apply Calibration", type="primary", use_container_width=True):
            # Create calibrated DataFrame
            calibrated_df = pd.DataFrame(calibrated_columns, index=df.index)
            
            # Add calibrated outcome
            try:
                calibrated_df["_OUTCOME_MEMBERSHIP"] = outcome_to_membership(df[outcome])
            except Exception as e:
                st.error(f"Error adding outcome membership: {str(e)}")
                calibrated_df["_OUTCOME_MEMBERSHIP"] = np.nan
            
            # Save to session state
            st.session_state["calibrated_df"] = calibrated_df
            st.session_state["calibration_config"] = calibration_config
            
            st.success("✅ Calibration applied successfully")
            st.balloons()
    
    with col_export:
        if st.button("📥 Export Configuration", use_container_width=True):
            json_str = json.dumps(calibration_config, indent=2, ensure_ascii=False)
            st.download_button(
                "Download JSON",
                json_str,
                "calibration_config.json",
                "application/json"
            )
    
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            keys_to_remove = ["calibrated_df", "calibration_config"]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Calibration reset")
            st.rerun()
    
    # Instructions to continue
    if "calibrated_df" in st.session_state:
        st.info("✅ Calibration ready. Continue with 'Truth Table'.")


# Direct execution
if __name__ == "__main__":
    show()