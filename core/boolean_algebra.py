# core/boolean_algebra.py
"""
Boolean Algebra Engine for QCA
------------------------------
This module handles the interpretation and evaluation of Boolean expressions
used in Qualitative Comparative Analysis (QCA). It supports:

• Crisp and fuzzy operators
• Logical AND, OR, NOT
• Parsing minimization output (e.g., "A*b + ~C")
• Evaluating membership scores of complex formulas
• Standardized functions used across QCA modules

Author: TFG Marla — People Analytics QCA Platform
"""

import re
import numpy as np
import pandas as pd

from .fuzzy_logic import negate, clip01


# =============================================================================
# BASIC OPERATORS
# =============================================================================

def fuzzy_and(a, b):
    """
    Fuzzy AND (minimum rule)
    """
    return np.minimum(a, b)


def fuzzy_or(a, b):
    """
    Fuzzy OR (maximum rule)
    """
    return np.maximum(a, b)


def fuzzy_not(a):
    """
    Fuzzy negation (~)
    """
    return 1 - a


# =============================================================================
# PARSING BOOLEAN EXPRESSIONS
# =============================================================================

def tokenize_expression(expr):
    """
    Takes an expression like:
        "A*b + ~C"
    Returns a clean list of tokens.

    Example:
        tokenize("A*b + ~C")
        → ['A', '*', 'b', '+', '~', 'C']
    """
    expr = expr.replace(" ", "")
    tokens = re.findall(r'~|[\+\*]|\w+', expr)
    return tokens


def split_into_terms(expr):
    """
    Split an expression like:
        "A*b + C*d + ~E"
    into:
        ["A*b", "C*d", "~E"]
    """
    expr = expr.replace(" ", "")
    return expr.split("+")


# =============================================================================
# EVALUATE A SINGLE TERM (e.g. "A*b", "~X*Y")
# =============================================================================

def evaluate_single_term(term, df):
    """
    Evaluates a Boolean term using fuzzy logic operators.
    """
    term = term.replace(" ", "")
    
    if not term:  # Handle empty term
        return np.ones(len(df))
    
    factors = term.split("*")
    
    # Start with 1.0
    result = np.ones(len(df), dtype=float)
    
    for f in factors:
        if f.startswith("~"):
            var = f[1:]
            if var not in df.columns:
                raise KeyError(f"Variable '{var}' not found in dataset.")
            values = 1.0 - df[var].astype(float)
        else:
            var = f
            if var not in df.columns:
                raise KeyError(f"Variable '{var}' not found in dataset.")
            values = df[var].astype(float)
        
        # Multiply (fuzzy AND = min, pero con multiplicación es más suave)
        result = np.minimum(result, values)
    
    return np.clip(result, 0.0, 1.0)


# =============================================================================
# EVALUATE FULL BOOLEAN EXPRESSION (e.g. "A*b + C*~D")
# =============================================================================

def evaluate_expression(expr, df):
    """
    Evaluates a complete Boolean expression containing
    AND (*), OR (+), and NOT (~).

    Example:
        expr = "A*b + C*~D"

    Returns:
        np.ndarray (fuzzy membership of the Boolean formula)
    """
    expr = expr.strip()
    terms = split_into_terms(expr)

    # Start with empty membership = 0 (OR aggregation)
    final = np.zeros(len(df))

    for t in terms:
        membership = evaluate_single_term(t, df)
        final = fuzzy_or(final, membership)

    return clip01(final)


# =============================================================================
# EXTRACT VARIABLE NAMES FROM EXPRESSION
# =============================================================================

def extract_variables(expr):
    """
    Extracts variable names from a Boolean expression.

    Example:
        "~A*B + c"
        → ['A', 'B', 'c']
    """
    tokens = tokenize_expression(expr)
    variables = [t.replace("~", "") for t in tokens if t not in ["~", "+", "*"]]
    return sorted(list(set(variables)))


# =============================================================================
# BOOLEAN FORMULA VALIDATION
# =============================================================================

def validate_expression(expr, df_columns):
    """
    Checks if a Boolean expression only contains valid variables.

    Returns:
        (True, None) if valid
        (False, "Error message") if invalid
    """
    vars_in_expr = extract_variables(expr)

    missing = [v for v in vars_in_expr if v not in df_columns]
    if missing:
        return False, f"Variables not found in dataset: {missing}"

    return True, None


# =============================================================================
# GENERATE FULL BOOLEAN MATRIX (all terms)
# =============================================================================

def build_boolean_matrix(expressions, df):
    """
    Given a list of expressions:
        ["A*b", "C*~D", "A + B*C"]

    Returns a DataFrame where each column is the fuzzy membership
    of the evaluated expression.
    """
    result = {}

    for expr in expressions:
        result[expr] = evaluate_expression(expr, df)

    return pd.DataFrame(result)


# =============================================================================
# BOOLEAN SIMPLIFICATION HELPERS (USED IN MINIMIZATION)
# =============================================================================

def normalize_term(term):
    """
    Sorts letters inside a term to canonical form.
    Positive first, then negated, alphabetical.
    """
    term = term.replace(" ", "")
    if not term:
        return ""
    
    factors = term.split("*")
    
    # Separate positive and negated
    positive = sorted([f for f in factors if not f.startswith("~")])
    negated = sorted([f for f in factors if f.startswith("~")])
    
    # Join: positive first, then negated
    return "*".join(positive + negated)
