# qca_engine.py
"""
QCA ENGINE — Motor de Minimización Booleana para QCA
TFG — Plataforma de Analítica Cualitativa Comparada

Produce:
    - Solución COMPLEJA (C)
    - Solución INTERMEDIA (I)
    - Solución PARSIMONIOSA (P)

Incluye:
    - Identificación de configuraciones esenciales
    - Gestión de remainder rows
    - Minimización via Quine–McCluskey
    - Cálculo de métricas por solución (Consistency, Coverage, PRI)
    - Ensamblaje de expresiones causales
"""

import pandas as pd
import numpy as np
import itertools
from functools import reduce

# ============================================================
# Helper utilities
# ============================================================

def term_to_string(term_dict):
    """
    Convierte un término booleano (dict) en formato textual:
    Ej: {"A":1, "B":0, "C":1} → A*~B*C
    """
    parts = []
    for k,v in term_dict.items():
        if v == 1:
            parts.append(k)
        elif v == 0:
            parts.append(f"~{k}")
    return "*".join(parts)

def evaluate_term(term_dict, df):
    """
    Evalúa un término contra una truth table binaria.
    Devuelve una serie booleana donde el término es verdadero.
    """
    mask = pd.Series([True] * len(df))
    for cond,val in term_dict.items():
        if val == 1:
            mask &= (df[cond] == 1)
        elif val == 0:
            mask &= (df[cond] == 0)
    return mask.astype(int)

def is_subset(d1, d2):
    """
    Verifica si dict d1 es subconjunto de d2.
    Ej: {"A":1} es subset de {"A":1,"B":0}
    """
    return all(item in d2.items() for item in d1.items())

# ============================================================
# Quine–McCluskey Minimization
# ============================================================

def combine_terms(t1, t2):
    """
    Combina dos términos booleanos si difieren en 1 literal.
    Ej: {A:1,B:0,C:1} + {A:1,B:1,C:1} → {A:1,B:'-',C:1}
    """
    new = {}
    diff = 0
    for k in t1:
        if t1[k] == t2[k]:
            new[k] = t1[k]
        else:
            new[k] = "-"
            diff += 1
    return new if diff == 1 else None

def qm_minimize(terms):
    """
    Procesa una lista de términos (dictionaries) y devuelve prime implicants.
    """
    uncombined = []
    combined = set()
    groups = {}
    
    # Group by number of 1s
    for t in terms:
        count = sum(1 for v in t.values() if v == 1)
        groups.setdefault(count, []).append(t)

    new_groups = {}
    used = set()

    # Try to combine neighboring groups
    for c in sorted(groups.keys()):
        if c + 1 not in groups:
            continue
        for t1 in groups[c]:
            for t2 in groups[c+1]:
                combined_term = combine_terms(t1, t2)
                if combined_term:
                    used.add(tuple(t1.items()))
                    used.add(tuple(t2.items()))
                    new_groups.setdefault(c, []).append(combined_term)

    # Uncombined → candidates for prime implicants
    for c in groups:
        for t in groups[c]:
            if tuple(t.items()) not in used:
                uncombined.append(t)

    # Recursive minimization on new_groups
    if new_groups:
        return uncombined + qm_minimize(new_groups[c] for c in new_groups)
    return uncombined

# ============================================================
# Main QCA Engine
# ============================================================

class QCAEngine:

    def __init__(self, truth_table, conditions, outcome_col="_OUTCOME_MEMBERSHIP"):
        self.tt = truth_table.copy()
        self.conditions = conditions
        self.outcome_col = outcome_col

        # Only use positive-consistent configurations
        self.tt_pos = self.tt[self.tt["OUT_mean"] > 0.5].reset_index(drop=True)

        # Build binary representation for QM
        self.binary_table = self._build_binary_table()

    def _build_binary_table(self):
        """
        Tabla binaria: cada condición → 0/1. Usada para QM.
        """
        bin_df = pd.DataFrame()
        for cond in self.conditions:
            # ya vienen crisp desde truth_table
            bin_df[cond] = (self.tt_pos[f"{cond}_avg"] >= 0.5).astype(int)
        return bin_df

    # --------------------------------------------------------
    #    GENERATE COMPLEX SOLUTION (C)
    # --------------------------------------------------------

    def complex_solution(self):
        """
        Solución COMPLEJA:
        - Solo configuraicones observadas
        - NO usa remainder rows
        - Minimización exacta del mapa empírico
        """
        terms = []
        for _, row in self.binary_table.iterrows():
            terms.append({cond: int(row[cond]) for cond in self.conditions})

        primes = qm_minimize(terms)
        solution_str = " + ".join(term_to_string(p) for p in primes)
        return {
            "type": "C",
            "terms": primes,
            "expression": solution_str,
            "metrics": self.evaluate_solution(primes)
        }

    # --------------------------------------------------------
    #    GENERATE PARSIMONIOUS SOLUTION (P)
    # --------------------------------------------------------

    def parsimonious_solution(self):
        """
        Solución PARSIMONIOSA:
        - Incluye remainder rows
        - Maximiza simplificación
        """
        # Build all possible configurations = 2^k combinations
        all_configs = []
        for combo in itertools.product([0,1], repeat=len(self.conditions)):
            term = {cond: val for cond,val in zip(self.conditions, combo)}
            all_configs.append(term)

        # Mark which configs are positive empirical
        positive_terms = [
            {cond: int(row[cond]) for cond in self.conditions}
            for _, row in self.binary_table.iterrows()
        ]

        remainder_rows = [t for t in all_configs if t not in positive_terms]
        terms_to_minimize = positive_terms + remainder_rows

        primes = qm_minimize(terms_to_minimize)
        solution_str = " + ".join(term_to_string(p) for p in primes)

        return {
            "type": "P",
            "terms": primes,
            "expression": solution_str,
            "metrics": self.evaluate_solution(primes)
        }

    # --------------------------------------------------------
    #    INTERMEDIATE SOLUTION (I)
    # --------------------------------------------------------

    def intermediate_solution(self, directional_expectations=None):
        """
        Intermedia = mezcla entre P y C pero restringida por
        supuestos teóricos: "directional expectations".
        Ej: {"A":1, "B":None, "C":0} significa:
            A favorece outcome, C no favorece, B libre.

        Si no se pasan directional expectations → usa compleja.
        """
        if directional_expectations is None:
            return self.complex_solution()

        P = self.parsimonious_solution()["terms"]
        C = self.complex_solution()["terms"]

        # Keep only primes consistent with directional expectations
        filtered = []
        for term in P:
            ok = True
            for cond,exp in directional_expectations.items():
                if exp is None:
                    continue
                if term.get(cond) != exp:
                    ok = False
                    break
            if ok:
                filtered.append(term)

        # Also allow empirical necessity from complex
        for term in C:
            if term not in filtered:
                filtered.append(term)

        solution_str = " + ".join(term_to_string(t) for t in filtered)

        return {
            "type": "I",
            "terms": filtered,
            "expression": solution_str,
            "metrics": self.evaluate_solution(filtered)
        }

    # --------------------------------------------------------
    #    METRIC EVALUATION
    # --------------------------------------------------------

    def evaluate_solution(self, terms):
        """
        Calcula Consistency, Coverage y PRI de toda la solución.
        """
        df = self.tt.copy()

        # Add column: solution membership (fuzzy OR over conj. fuzzy)
        membership = np.zeros(len(df))
        for term in terms:
            mask = np.ones(len(df))
            for cond,val in term.items():
                if val == 1:
                    mask *= df[f"{cond}_avg"]
                elif val == 0:
                    mask *= (1 - df[f"{cond}_avg"])
            membership = np.maximum(membership, mask)

        Y = df["OUT_mean"].values

        # Consistency = Σ min(solution, Y) / Σ solution
        numer = np.minimum(membership, Y).sum()
        denom = membership.sum()
        consistency = numer / denom if denom > 0 else np.nan

        # Coverage = Σ min(solution, Y) / Σ Y
        denom2 = Y.sum()
        coverage = numer / denom2 if denom2 > 0 else np.nan

        # PRI = Σ min(solution,Y) / ( Σ min(solution,Y) + Σ min(solution,~Y) )
        numer_pri = np.minimum(membership, Y).sum()
        denom_pri = numer_pri + np.minimum(membership, 1-Y).sum()
        pri = numer_pri / denom_pri if denom_pri > 0 else np.nan

        return {
            "Consistency": consistency,
            "Coverage": coverage,
            "PRI": pri
        }
