# app.py
"""
Main app router for TFG People Analytics + QCA
"""

import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

# Page config
st.set_page_config(
    page_title="TFG People Analytics — QCA Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import modules (with error handling)
def safe_import(module_name):
    """Safely import a module."""
    try:
        if module_name == "start":
            import start
            return start
        elif module_name == "calibration":
            import modules.calibration as calibration
            return calibration
        elif module_name == "truth_table":
            import modules.truth_table as truth_table
            return truth_table
        elif module_name == "qca_results":
            import modules.qca_results as qca_results
            return qca_results
        elif module_name == "configuration_visualizer":
            import modules.configuration_visualizer as configuration_visualizer
            return configuration_visualizer
        elif module_name == "robustness_check":
            import modules.robustness_check as robustness_check
            return robustness_check
        elif module_name == "report_generator":
            import modules.report_generator as report_generator
            return report_generator
        elif module_name == "about":
            import about
            return about
    except ImportError as e:
        st.sidebar.error(f"Module {module_name} not found: {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading {module_name}: {e}")
        return None

# Header
def draw_header():
    st.markdown("""
    <div style="background-color:#283044;padding:20px;border-radius:10px;margin-bottom:20px">
        <h1 style="color:white;margin:0">TFG — People Analytics · QCA Platform</h1>
        <p style="color:#cccccc;margin:5px 0 0 0">
            Interface for organizational diagnostics, set-theoretic calibration, QCA minimization
        </p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "raw_df": None,
        "conditions": None,
        "outcome": None,
        "calibrated_df": None,
        "truth_table": None,
        "truth_table_filtered": None,
        "qca_solutions": None,
        "ready_for_analysis": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Start",
            "Calibration", 
            "Truth Table",
            "QCA Solutions",
            "Visualizer",
            "Robustness",
            "Reports",
            "About"
        ],
        icons=[
            "house",
            "sliders",
            "table",
            "gear",
            "map",
            "shield-check",
            "file-text",
            "info-circle"
        ],
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "nav-link": {"font-size": "14px", "margin": "2px"},
            "nav-link-selected": {"background-color": "#283044"},
        }
    )

# Initialize
init_session_state()
draw_header()

# Routing
if selected == "Start":
    module = safe_import("start")
    if module:
        module.show()
    else:
        st.error("Module 'start' not available. Create start.py file")

elif selected == "Calibration":
    module = safe_import("calibration")
    if module:
        module.show()
    else:
        st.error("Module 'calibration' not available")

elif selected == "Truth Table":
    module = safe_import("truth_table")
    if module:
        module.show()
    else:
        st.error("Module 'truth_table' not available")

elif selected == "QCA Solutions":
    module = safe_import("qca_results")
    if module:
        module.show()
    else:
        st.error("Module 'qca_results' not available")

elif selected == "Visualizer":
    module = safe_import("configuration_visualizer")
    if module:
        module.show()
    else:
        st.error("Module 'configuration_visualizer' not available")

elif selected == "Robustness":
    module = safe_import("robustness_check")
    if module:
        module.show()
    else:
        st.error("Module 'robustness_check' not available")

elif selected == "Reports":
    module = safe_import("report_generator")
    if module:
        module.show()
    else:
        st.error("Module 'report_generator' not available")

elif selected == "About":
    module = safe_import("about")
    if module:
        module.show()
    else:
        st.info("Create about.py for project information")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px'>"
    "TFG People Analytics · QCA Platform · © 2024"
    "</div>",
    unsafe_allow_html=True
)