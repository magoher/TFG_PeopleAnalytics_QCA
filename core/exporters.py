# core/exporters.py
"""
Export Utilities for QCA Platform
---------------------------------
This module centralizes all export functions used in the
People Analytics QCA TFG platform.

Designed for:
• Managers & executives (clear summaries)
• Researchers (JSON + CSV + Markdown)
• Reproducibility (ZIP bundles)

Exports supported:
• JSON
• CSV
• Excel Workbook (.xlsx)
• Markdown Report (.md)
• ZIP archive

Author: TFG Marla — People Analytics QCA Platform
"""

import os
import json
import zipfile
import pandas as pd
from datetime import datetime


# ======================================================
# INTERNAL UTILS
# ======================================================

def _timestamp():
    """Returns timestamp for file naming."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path):
    """Ensures an output directory exists."""
    os.makedirs(path, exist_ok=True)


# ======================================================
# JSON EXPORT
# ======================================================

def export_json(data, output_path, name="qca_export"):
    """
    Exports any Python object to JSON (solutions, TT, calibration, etc.).

    Parameters
    ----------
    data : dict or list
    output_path : str
    name : str
    """
    _ensure_dir(output_path)
    filename = os.path.join(output_path, f"{name}_{_timestamp()}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return filename


# ======================================================
# CSV EXPORT
# ======================================================

def export_csv(df, output_path, name="qca_export"):
    """
    Saves a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    output_path : str
    name : str
    """
    _ensure_dir(output_path)
    filename = os.path.join(output_path, f"{name}_{_timestamp()}.csv")
    df.to_csv(filename, index=False)
    return filename


# ======================================================
# EXCEL EXPORT (MANAGER-FRIENDLY)
# ======================================================

def export_excel(sheets_dict, output_path, name="qca_report"):
    """
    Exports multiple DataFrames into a single Excel workbook.

    Perfect for managers who want the entire analysis in one place.

    Parameters
    ----------
    sheets_dict : dict
        { "SheetName": DataFrame }
    output_path : str
    name : str
    """
    _ensure_dir(output_path)
    filename = os.path.join(output_path, f"{name}_{_timestamp()}.xlsx")

    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return filename


# ======================================================
# MARKDOWN EXPORT (EXECUTIVE SUMMARY)
# ======================================================

def export_markdown(summary_dict, output_path, name="qca_summary"):
    """
    Creates an executive-style markdown report summarizing
    key QCA results in a clean, manager-friendly format.

    Parameters
    ----------
    summary_dict : dict
        Should contain keys like:
        {
            "title": str,
            "conditions": [...],
            "solution_terms": [...],
            "metrics": {...},
            "interpretation": str
        }
    """
    _ensure_dir(output_path)
    filename = os.path.join(output_path, f"{name}_{_timestamp()}.md")

    lines = []

    # Title
    lines.append(f"# {summary_dict.get('title', 'QCA Report')}\n")

    # Conditions
    if "conditions" in summary_dict:
        lines.append("## Conditions Used\n")
        for c in summary_dict["conditions"]:
            lines.append(f"- **{c}**")
        lines.append("")

    # Solution Terms
    if "solution_terms" in summary_dict:
        lines.append("## Configurations Leading to the Outcome\n")
        for term in summary_dict["solution_terms"]:
            lines.append(f"- `{term}`")
        lines.append("")

    # Metrics
    if "metrics" in summary_dict:
        lines.append("## Key Metrics\n")
        for k, v in summary_dict["metrics"].items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Interpretation
    if "interpretation" in summary_dict:
        lines.append("## Interpretation Summary\n")
        lines.append(summary_dict["interpretation"])
        lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filename


# ======================================================
# ZIP EXPORT (FULL BUNDLE)
# ======================================================

def export_zip(file_list, output_path, name="qca_bundle"):
    """
    Creates a ZIP archive containing multiple generated files.

    Parameters
    ----------
    file_list : list[str]
        Paths to files already exported
    output_path : str
    name : str
    """
    _ensure_dir(output_path)
    filename = os.path.join(output_path, f"{name}_{_timestamp()}.zip")

    with zipfile.ZipFile(filename, "w") as zipf:
        for file in file_list:
            if os.path.exists(file):
                zipf.write(file, arcname=os.path.basename(file))

    return filename


# ======================================================
# HIGH-LEVEL MANAGER EXPORTER
# ======================================================

def export_full_manager_package(
    truth_table=None,
    solutions=None,
    metrics=None,
    interpretation="",
    output_path="exports"
):
    """
    Export a full, manager-level QCA package:
    • Truth table (CSV)
    • Solutions (JSON)
    • Metrics (JSON)
    • Executive summary (Markdown)
    • Excel workbook consolidating everything
    • ZIP bundle containing all files

    Returns
    -------
    dict with all paths
    """
    files = {}

    # CSV truth table
    if truth_table is not None:
        files["truth_table_csv"] = export_csv(
            truth_table, output_path, "truth_table"
        )

    # JSON solutions
    if solutions is not None:
        files["solutions_json"] = export_json(
            solutions, output_path, "qca_solutions"
        )

    # JSON metrics
    if metrics is not None:
        files["metrics_json"] = export_json(
            metrics, output_path, "qca_metrics"
        )

    # Markdown managerial summary
    summary = {
        "title": "QCA Executive Summary",
        "conditions": truth_table.columns[:-1].tolist()
        if truth_table is not None else [],
        "solution_terms": solutions.get("prime_implicants", [])
        if solutions else [],
        "metrics": metrics if metrics else {},
        "interpretation": interpretation
    }

    files["markdown_summary"] = export_markdown(
        summary, output_path, "executive_summary"
    )

    # Excel workbook
    excel_data = {}
    if truth_table is not None:
        excel_data["Truth Table"] = truth_table
    if solutions:
        excel_data["Solutions"] = pd.DataFrame(solutions)
    if metrics:
        excel_data["Metrics"] = pd.DataFrame(metrics, index=[0])

    files["excel_report"] = export_excel(
        excel_data, output_path, "full_qca_report"
    )

    # ZIP bundle
    files["zip_bundle"] = export_zip(
        list(files.values()), output_path, "qca_full_package"
    )

    return files
