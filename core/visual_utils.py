# core/visual_utils.py
"""
Visual Utilities for QCA
------------------------
This module provides reusable visual-building utilities for the 
People Analytics QCA platform. It does *not* render visuals directly 
through Streamlit — instead, it constructs clean intermediate 
representations that the UI layer can plot.

Functions provided:
• build_node_graph()          → causal diagrams, term graphs
• build_configuration_map()   → case–condition matrix for heatmaps
• build_venn_structure()      → Venn diagram logical sets
• parse_boolean_expression()  → string → structured term representation
• expand_term()               → convert “A*~B*C” → dict form
• stringify_term()            → reverse operation

Author: TFG Marla — People Analytics QCA Platform
"""

import itertools
import numpy as np
import pandas as pd


# ======================================================
# BOOLEAN TERM PARSING
# ======================================================

def parse_boolean_expression(expr):
    """
    Parses a minimized boolean expression string into individual terms.

    Example:
        "A*B + ~C*D" 
    Returns:
        ["A*B", "~C*D"]

    Safe low-level parser — does not evaluate anything.
    """
    if expr is None or str(expr).strip() == "":
        return []

    return [t.strip() for t in expr.replace(" ", "").split("+")]


def expand_term(term):
    """
    Converts a single term like:
        "A*~B*C"
    Into a dictionary structure:
        { "A": 1, "B": 0, "C": 1 }

    Returns:
        dict(condition → membership {1,0})
    """
    factors = term.split("*")
    parsed = {}

    for f in factors:
        if f.startswith("~"):
            parsed[f[1:]] = 0
        else:
            parsed[f] = 1

    return parsed


def stringify_term(term_dict):
    """
    Converts a dict term back into a readable expression.

    Example:
        { "A": 1, "B": 0, "C": 1 }
    Result:
        "A*~B*C"
    """
    parts = []
    for k, v in term_dict.items():
        if v == 1:
            parts.append(k)
        else:
            parts.append(f"~{k}")
    return "*".join(parts)


# ======================================================
# NODE-LINK GRAPH STRUCTURE
# ======================================================

def build_node_graph(terms, outcome="Y"):
    """
    Creates a node–edge structure for causal diagrams.

    Parameters
    ----------
    terms : list[str]
        List of boolean-minimized expressions.
        e.g., ["A*B", "~C*D"]

    Returns
    -------
    dict:
        {
            "nodes": [
                {"id": "A"}, {"id": "B"}, ..., {"id": "Y"}
            ],
            "links": [
                {"source": "A", "target": "Y"},
                {"source": "B", "target": "Y"},
                ...
            ]
        }

    Notes:
    • Terms are expanded automatically.
    • Outcome is automatically added as final node.
    """
    node_set = set()
    edges = []

    for term in terms:
        parsed = expand_term(term)
        for cond in parsed.keys():
            node_set.add(cond)
            edges.append({"source": cond, "target": outcome})

    node_set.add(outcome)

    nodes = [{"id": n} for n in node_set]

    return {"nodes": nodes, "links": edges}


# ======================================================
# CONFIGURATION HEATMAP DATA
# ======================================================

def build_configuration_map(df, conditions, outcome):
    """
    Builds a matrix suited for heatmap visualizations.

    Given a truth table-like structure:
        df[conditions] = binary or fuzzy sets
        df[outcome]    = binary/fuzzy Y

    Returns:
        dict with:
        - "matrix" → numeric values (cases × conditions)
        - "row_labels"
        - "col_labels"
        - "outcome_vector"

    Perfect for:
    • Configuration heatmaps
    • Truth table visualization
    • Condition coverage exploration
    """
    matrix = df[conditions].to_numpy()
    outcome_vec = df[outcome].to_numpy()

    return {
        "matrix": matrix,
        "row_labels": df.index.astype(str).tolist(),
        "col_labels": conditions,
        "outcome_vector": outcome_vec
    }


# ======================================================
# VENN DIAGRAM LOGIC MODEL
# ======================================================

def build_venn_structure(conditions):
    """
    Builds Venn regions for conditions.

    Example:
        ["A", "B", "C"]

    Returns:
        all combinations:
        [
            {"set": ("A",), "label": "A"},
            {"set": ("B",), ...}
            {"set": ("A","B"), ...}
            {"set": ("A","C"), ...}
            {"set": ("A","B","C"), ...}
        ]

    This does not compute areas — it only computes *logical regions*.
    """
    regions = []

    for r in range(1, len(conditions) + 1):
        combos = itertools.combinations(conditions, r)
        for c in combos:
            regions.append({
                "set": tuple(c),
                "label": "*".join(c)
            })

    return regions


# ======================================================
# IMPLICANT → CASE COVERAGE MAP
# ======================================================

def build_case_coverage(df, terms, conditions):
    """
    Maps each term (e.g., A*B, ~C*D) to the cases that satisfy it.

    Returns:
        dict:
            {
                "A*B": ["case1", "case7"],
                "~C*D": ["case2", "case5", "case9"]
            }
    """

    coverage = {}

    for term in terms:
        parsed = expand_term(term)
        mask = pd.Series([True] * len(df))

        # build boolean filter
        for cond, expected in parsed.items():
            if expected == 1:
                mask &= df[cond] >= 0.5      # fuzzy membership threshold
            else:
                mask &= df[cond] < 0.5

        coverage[term] = df[mask].index.astype(str).tolist()

    return coverage


# ======================================================
# CONFIGURATION DESCRIPTOR
# ======================================================

def describe_configuration(term, coverage_list):
    """
    Builds a small metadata block describing a configuration.

    Parameters
    ----------
    term : str
    coverage_list : list[str]

    Returns:
        dict with:
        - term
        - readable_label
        - case_count
        - cases
    """
    parsed = expand_term(term)

    label_parts = []
    for k, v in parsed.items():
        if v == 1:
            label_parts.append(k)
        else:
            label_parts.append(f"NOT {k}")

    readable = " AND ".join(label_parts)

    return {
        "term": term,
        "readable_label": readable,
        "case_count": len(coverage_list),
        "cases": coverage_list
    }
