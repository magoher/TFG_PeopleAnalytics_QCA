# core/qca_engine.py
"""
QCA Engine
----------------------------------

A robust QCA engine:
 - Computes Complex (C), Parsimonious (P), Intermediate (I) solutions
 - Works with fuzzy or crisp truth tables
 - Computes Consistency, Coverage, PRI using fuzzy logic
 - Uses Minimizer when available, otherwise graceful fallback
 - Saves solutions to runtime/qca_solutions.json

"""

import os
import json
import itertools
import math
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd




# -------------------------
# Attempt to import helpers (safe)
# -------------------------
try:
    from .fuzzy_logic import clip01, negate, is_fuzzy, calibrate_fuzzy
except Exception:
    # minimal fallbacks if fuzzy_logic is missing
    def clip01(x):
        return np.clip(x, 0, 1)

    def negate(x):
        return 1 - np.array(x, dtype=float)

    def is_fuzzy(s):
        s_series = pd.Series(s)
        return (s_series.min() >= 0) and (s_series.max() <= 1)

    def calibrate_fuzzy(series, thresholds=None, method="three", steepness=1.5):
        s = pd.Series(series).astype(float)
        denom = float(max((s.max() - s.min()), 1e-9))
        return ((s - s.min()) / denom).to_numpy()


try:
    from .minimization import Minimizer
except Exception:
    Minimizer = None  # engine will fallback if minimizer missing


try:
    from .exporters import export_json, _ensure_dir
except Exception:
    def export_json(data, output_path, name="qca_solutions"):
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)


# -------------------------
# Runtime constants
# -------------------------
RUNTIME_DIR = "runtime"
SOLUTIONS_FILE = os.path.join(RUNTIME_DIR, "qca_solutions.json")


# -------------------------
# Internal helpers
# -------------------------
def _resolve_column(df: pd.DataFrame, cond: str) -> str:
    """Prefer '<cond>_avg' if it exists, otherwise '<cond>'."""
    candidate = f"{cond}_avg"
    if candidate in df.columns:
        return candidate
    if cond in df.columns:
        return cond
    raise KeyError(f"Condition column not found for '{cond}' (tried '{candidate}', '{cond}').")


def _as_membership_array(df: pd.DataFrame, cond: str) -> np.ndarray:
    col = _resolve_column(df, cond)
    return clip01(df[col].astype(float).to_numpy())


def save_solutions_to_runtime(data: Dict[str, Any], path: str = SOLUTIONS_FILE) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_solutions_from_runtime(path: str = SOLUTIONS_FILE) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# QCAEngine class
# =========================================================
class QCAEngine:
    def __init__(
        self,
        truth_table: pd.DataFrame,
        conditions: List[str],
        outcome_col: str = "OUT_mean",
        freq_col: Optional[str] = None,
        consistency_cut: float = 0.8,
        freq_cut: int = 1,
        save_runtime: bool = True,
    ) -> None:
        """
        Initialize QCA engine.

        - truth_table: DataFrame with observed configurations and membership stats
        - conditions: list of base condition names (without '_avg')
        - outcome_col: name of outcome membership column (default 'OUT_mean')
        - freq_col: optional frequency column name ('n' or 'N_cases' typical)
        """
        if truth_table is None or truth_table.empty:
            raise ValueError("truth_table must be a non-empty DataFrame.")

        self.tt = truth_table.copy().reset_index(drop=True)
        self.conditions = list(conditions)
        self.outcome_col = outcome_col

        # decide frequency column
        if freq_col:
            self.freq_col = freq_col
        elif "n" in self.tt.columns:
            self.freq_col = "n"
        elif "N_cases" in self.tt.columns:
            self.freq_col = "N_cases"
        else:
            self.freq_col = None

        self.consistency_cut = consistency_cut
        self.freq_cut = freq_cut
        self.save_runtime = save_runtime

        if self.outcome_col not in self.tt.columns:
            raise KeyError(f"Outcome column '{self.outcome_col}' not found in truth_table.")

        # Build membership DataFrame (one column per condition)
        self._membership_df = pd.DataFrame()
        for c in self.conditions:
            try:
                self._membership_df[c] = _as_membership_array(self.tt, c)
            except KeyError:
                if c in self.tt.columns:
                    self._membership_df[c] = clip01(self.tt[c].astype(float).to_numpy())
                else:
                    raise

        # outcome vector
        self._outcome = clip01(self.tt[self.outcome_col].astype(float).to_numpy())

        # frequency vector
        if self.freq_col and self.freq_col in self.tt.columns:
            self._freq = self.tt[self.freq_col].astype(int).to_numpy()
        else:
            self._freq = np.ones(len(self.tt), dtype=int)

        # crisp binary matrix (threshold 0.5)
        self._binary_df = (self._membership_df >= 0.5).astype(int)

        # container for computed solutions
        self.solutions: Dict[str, Any] = {}

    # -------------------------
    # High-level pipeline
    # -------------------------
    def compute_all_solutions(self, directional_expectations: Optional[Dict[str, Optional[int]]] = None) -> Dict[str, Any]:
        """
        Compute Complex, Parsimonious and Intermediate solutions.
        directional_expectations example: {'A':1, 'B':0, 'C': None}
        """
        complex_res = self.complex_solution()
        parsimonious_res = self.parsimonious_solution()
        intermediate_res = self.intermediate_solution(directional_expectations)

        out = {
            "Complex": complex_res,
            "Parsimonious": parsimonious_res,
            "Intermediate": intermediate_res,
            "meta": {
                "conditions": self.conditions,
                "outcome_col": self.outcome_col,
                "consistency_cut": self.consistency_cut,
                "freq_cut": self.freq_cut,
                "rows": int(self.tt.shape[0]),
            }
        }

        if self.save_runtime:
            try:
                save_solutions_to_runtime(out)
            except Exception as exc:
                out["_save_error"] = str(exc)

        self.solutions = out
        return out

    # -------------------------
    # Complex solution
    # -------------------------
    def complex_solution(self) -> Dict[str, Any]:
        """Use empirically observed positive rows (OUT >= 0.5)."""
        mask_pos = self._outcome >= 0.5
        df_pos = self._membership_df[mask_pos]

        if df_pos.empty:
            return {"type": "C", "terms": [], "expression": "", "metrics": {}}

        terms: List[Dict[str, int]] = []
        for _, row in df_pos.iterrows():
            terms.append({c: int(row[c] >= 0.5) for c in self.conditions})

        expression = " + ".join(["*".join([("" if v == 1 else "~") + k for k, v in t.items()]) for t in terms])
        metrics = self.evaluate_solution(terms)
        return {"type": "C", "terms": terms, "expression": expression, "metrics": metrics}

    # -------------------------
    # Parsimonious solution
    # -------------------------
    def parsimonious_solution(self) -> Dict[str, Any]:
        """
        Parsimonious solution: fallback uses unique observed binary configs.
        If Minimizer is available, you can adapt this method to delegate.
        """
        configs = [{c: int(row[c]) for c in self.conditions} for _, row in self._binary_df.iterrows()]

        # if Minimizer available, attempt to use it
        if Minimizer is not None:
            try:
                # Build a minimal truth-table-like DataFrame for minimizer
                df_configs = pd.DataFrame({
                    "configuration": ["*".join([("" if v == 1 else "~") + k for k, v in cfg.items()]) for cfg in configs],
                    "n": [int(self._freq[i]) for i in range(len(configs))],
                    "consistency": [float(self._outcome[i]) for i in range(len(configs))],
                    "outcome": [1 if self._outcome[i] >= 0.5 else 0 for i in range(len(configs))]
                })
                minim = Minimizer(df_configs, outcome_col="outcome", consistency_cut=self.consistency_cut, freq_cut=self.freq_cut)
                primes = minim.parsimonious_solution()
                # ensure primes are in dict form
                terms = []
                for p in primes:
                    if isinstance(p, str):
                        # parse "A*~B" to dict
                        d = {}
                        for lit in p.split("*"):
                            lit = lit.strip()
                            if lit.startswith("~"):
                                d[lit[1:]] = 0
                            else:
                                d[lit] = 1
                        terms.append(d)
                    elif isinstance(p, dict):
                        terms.append(p)
                expression = " + ".join(["*".join([("" if v == 1 else "~") + k for k, v in t.items()]) for t in terms])
                metrics = self.evaluate_solution(terms)
                return {"type": "P", "terms": terms, "expression": expression, "metrics": metrics}
            except Exception:
                # fallback to coarse method below
                pass

        # fallback: unique observed configs
        primes = []
        seen = set()
        for cfg in configs:
            tpl = tuple((k, cfg[k]) for k in sorted(cfg.keys()))
            if tpl not in seen:
                seen.add(tpl)
                primes.append(cfg)

        expression = " + ".join(["*".join([("" if v == 1 else "~") + k for k, v in t.items()]) for t in primes])
        metrics = self.evaluate_solution(primes)
        return {"type": "P", "terms": primes, "expression": expression, "metrics": metrics}

    # -------------------------
    # Intermediate solution
    # -------------------------
    def intermediate_solution(self, directional_expectations: Optional[Dict[str, Optional[int]]] = None) -> Dict[str, Any]:
        """
        Intermediate solution: filter parsimonious primes according to directional expectations.
        """
        p = self.parsimonious_solution()
        primes = p.get("terms", []) if isinstance(p.get("terms", []), list) else []

        if not directional_expectations:
            return {"type": "I", "terms": primes, "expression": p.get("expression", ""), "metrics": p.get("metrics", {})}

        filtered: List[Dict[str, int]] = []
        for t in primes:
            # normalize to dict
            term_dict = t if isinstance(t, dict) else {
                (lit[1:] if lit.startswith("~") else lit): (0 if lit.startswith("~") else 1)
                for lit in str(t).split("*") if lit.strip()
            }
            ok = True
            for cond, exp in directional_expectations.items():
                if exp is None:
                    continue
                if cond in term_dict and term_dict[cond] != exp:
                    ok = False
                    break
            if ok:
                filtered.append(term_dict)

        # ensure complex essentials present
        complex_terms = self.complex_solution().get("terms", [])
        for ct in complex_terms:
            if ct not in filtered:
                filtered.append(ct)

        expression = " + ".join(["*".join([("" if v == 1 else "~") + k for k, v in t.items()]) for t in filtered])
        metrics = self.evaluate_solution(filtered)
        return {"type": "I", "terms": filtered, "expression": expression, "metrics": metrics}

    # -------------------------
    # Evaluate solution metrics (fuzzy)
    # -------------------------
    def evaluate_solution(self, terms: List[Dict[str, int]]) -> Dict[str, float]:
        """
        Compute Consistency, Coverage and PRI for a given solution (list of terms).
        Terms may be dicts {cond:0/1} or strings "A*~B*C".
        """
        normalized_terms: List[Dict[str, int]] = []
        for t in terms:
            if isinstance(t, dict):
                normalized_terms.append(t)
            else:
                d: Dict[str, int] = {}
                for lit in str(t).split("*"):
                    lit = lit.strip()
                    if not lit:
                        continue
                    if lit.startswith("~"):
                        d[lit[1:]] = 0
                    else:
                        d[lit] = 1
                normalized_terms.append(d)

        # solution membership: OR across terms; term membership = AND across literals (min)
        solution_membership = np.zeros(len(self._membership_df), dtype=float)
        for term in normalized_terms:
            term_mem = np.ones(len(self._membership_df), dtype=float)
            for cond, val in term.items():
                vec = self._membership_df[cond].astype(float).to_numpy()
                if val == 1:
                    term_mem = np.minimum(term_mem, vec)
                else:
                    term_mem = np.minimum(term_mem, 1.0 - vec)
            solution_membership = np.maximum(solution_membership, term_mem)

        Y = self._outcome.astype(float)

        numer = np.minimum(solution_membership, Y).sum()
        denom_solution = solution_membership.sum()
        denom_y = Y.sum()

        consistency = float(numer / denom_solution) if denom_solution > 0 else float("nan")
        coverage = float(numer / denom_y) if denom_y > 0 else float("nan")
        denom_pri = numer + np.minimum(solution_membership, 1.0 - Y).sum()
        pri = float(numer / denom_pri) if denom_pri > 0 else float("nan")

        return {"Consistency": consistency, "Coverage": coverage, "PRI": pri}

# Añade esto al final de qca_engine.py, justo antes del final del archivo

def term_to_string(term, condition_names=None):
    """
    Convert a term dictionary to string representation.
    
    Parameters
    ----------
    term : dict
        Dictionary where keys are condition indices or names and values are 0/1
    condition_names : list, optional
        List of condition names for readable output
        
    Returns
    -------
    str
        String representation like "A*~B*C" or "Liderazgo*~Salario*Cultura"
    """
    if isinstance(term, str):
        return term  # Ya es string
    
    if not term:
        return ""
    
    parts = []
    for key, value in term.items():
        if condition_names and isinstance(key, int) and key < len(condition_names):
            name = condition_names[key]
        elif condition_names and key in condition_names:
            name = key
        else:
            # Si no hay nombres o key no es índice, usar letras
            if isinstance(key, int):
                name = chr(65 + key)  # A, B, C, ...
            else:
                name = str(key)
        
        if value == 1:
            parts.append(name)
        elif value == 0:
            parts.append(f"~{name}")
        # value == -1 or None means "don't care" (not included)
    
    return "*".join(parts)


# Y también modifica la exportación al final del archivo:
__all__ = ["QCAEngine", "term_to_string", "save_solutions_to_runtime", "load_solutions_from_runtime"]