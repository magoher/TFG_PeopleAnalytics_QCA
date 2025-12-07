# core/minimization.py
"""
Minimization Module for QCA - Simplified Wrapper
------------------------------------------------
This module provides a clean interface to the QCA minimization
engine implemented in qca_engine.py.

It acts as a facade/wrapper that:
1. Uses the existing QCAEngine (Quine-McCluskey implementation)
2. Provides simplified methods for complex/parsimonious/intermediate solutions
3. Formats results for easy consumption by other modules

Author: TFG Marla — People Analytics QCA Platform
"""

from .qca_engine import QCAEngine, term_to_string


class Minimizer:

    def __init__(self, truth_table, conditions=None, outcome_col="_OUTCOME_MEMBERSHIP", 
             consistency_cut=0.8, freq_cut=1):
        """
        Initialize the minimizer with truth table and conditions.
        """
        self.truth_table = truth_table
        self.outcome_col = outcome_col
        self.consistency_cut = consistency_cut
        self.freq_cut = freq_cut
        
        # If conditions not provided, infer from truth table
        if conditions is None:
            # Exclude common non-condition columns
            exclude_cols = {outcome_col, 'n', 'N_cases', 'consistency', 'outcome', 
                        'frequency', 'freq', 'config', 'configuration'}
            self.conditions = [col for col in truth_table.columns 
                            if col not in exclude_cols and not col.startswith('_')]
            
            # If still no conditions found, use placeholder
            if not self.conditions:
                self.conditions = ['A', 'B', 'C', 'D'][:min(4, len(truth_table.columns))]
        else:
            self.conditions = conditions
        
        # Initialize QCA engine
        self.engine = QCAEngine(
            truth_table, 
            self.conditions, 
            outcome_col=outcome_col,
            consistency_cut=consistency_cut,
            freq_cut=freq_cut
        )

    def complex_solution(self, as_strings=True):
        """
        Get the complex solution (only observed configurations).
        """
        solution = self.engine.complex_solution()
        
        if as_strings:
            return [term_to_string(t, self.conditions) for t in solution.get("terms", [])]
        return solution

    def parsimonious_solution(self, as_strings=True):
        """
        Get the parsimonious solution (with logical remainders).
        """
        solution = self.engine.parsimonious_solution()
        
        if as_strings:
            return [term_to_string(t, self.conditions) for t in solution.get("terms", [])]
        return solution

    def intermediate_solution(self, expectations=None, as_strings=True):
        """
        Get the intermediate solution (with directional expectations).
        """
        solution = self.engine.intermediate_solution(expectations)
        
        if as_strings:
            return [term_to_string(t, self.conditions) for t in solution.get("terms", [])]
        return solution

    def get_solution_metrics(self, solution_type="complex"):
        """
        Get metrics (consistency, coverage, PRI) for a solution type.
        """
        solution_map = {
            "complex": self.engine.complex_solution(),
            "parsimonious": self.engine.parsimonious_solution(),
            "intermediate": self.engine.intermediate_solution(),
        }
        
        solution = solution_map.get(solution_type.lower())
        if solution:
            return solution.get("metrics", {})
        return {}

    def summarize(self, include_metrics=True, include_objects=False):
        """
        Get a comprehensive summary of all solution types.
        """
        summary = {
            "complex_terms": self.complex_solution(as_strings=True),
            "parsimonious_terms": self.parsimonious_solution(as_strings=True),
            "intermediate_terms": self.intermediate_solution(as_strings=True),
        }
        
        if include_metrics:
            summary["complex_metrics"] = self.get_solution_metrics("complex")
            summary["parsimonious_metrics"] = self.get_solution_metrics("parsimonious")
            summary["intermediate_metrics"] = self.get_solution_metrics("intermediate")
        
        if include_objects:
            summary["complex_object"] = self.complex_solution(as_strings=False)
            summary["parsimonious_object"] = self.parsimonious_solution(as_strings=False)
            summary["intermediate_object"] = self.intermediate_solution(as_strings=False)
        
        return summary

    def get_all_prime_implicants(self):
        """
        Get all prime implicants from the parsimonious solution.
        """
        return self.parsimonious_solution(as_strings=True)

    def evaluate_solution_coverage(self, solution_terms, data=None):
        """
        Evaluate how many cases are covered by a solution.
        """
        if data is None:
            data = self.truth_table
        
        from .boolean_algebra import evaluate_expression
        
        # Combine all terms with OR
        if not solution_terms:
            return {"cases_covered": 0, "coverage_percentage": 0.0}
        
        # Create OR expression from all terms
        expression = " + ".join(solution_terms)
        
        # Evaluate membership
        membership = evaluate_expression(expression, data)
        
        # Calculate coverage
        n_cases = len(data)
        n_covered = (membership > 0.5).sum()
        coverage_pct = (n_covered / n_cases * 100) if n_cases > 0 else 0.0
        
        return {
            "cases_covered": int(n_covered),
            "total_cases": n_cases,
            "coverage_percentage": round(coverage_pct, 2),
            "average_membership": round(float(membership.mean()), 3)
        }

    


# =============================================================================
# Helper Functions
# =============================================================================

def create_minimizer_from_session_state():
    """
    Convenience function to create Minimizer from Streamlit session state.
    
    Looks for:
    - st.session_state["truth_table_filtered"]
    - st.session_state["conditions"]
    
    Returns
    -------
    Minimizer or None
        Minimizer instance if data is available, None otherwise
    """
    import streamlit as st
    
    if ("truth_table_filtered" not in st.session_state or 
        "conditions" not in st.session_state):
        return None
    
    truth_table = st.session_state["truth_table_filtered"]
    conditions = st.session_state["conditions"]
    
    if truth_table.empty or not conditions:
        return None
    
    return Minimizer(truth_table, conditions)


def format_solution_for_display(solution_terms, solution_type=""):
    """
    Format solution terms for nice display in Streamlit.
    
    Parameters
    ----------
    solution_terms : list
        List of solution terms as strings
    solution_type : str
        Type of solution (for header)
        
    Returns
    -------
    str
        Formatted string for display
    """
    if not solution_terms:
        return f"**{solution_type} Solution**: No terms found."
    
    header = f"**{solution_type} Solution**\n\n" if solution_type else ""
    
    # Format each term
    formatted_terms = []
    for i, term in enumerate(solution_terms, 1):
        # Replace * with AND and make it nicer
        readable = term.replace("*", " AND ").replace("~", "NOT ")
        formatted_terms.append(f"{i}. {readable}")
    
    return header + "\n".join(formatted_terms)


def compare_solutions(minimizer, solution1_type="complex", solution2_type="parsimonious"):
    """
    Compare two solution types.
    
    Parameters
    ----------
    minimizer : Minimizer
        Minimizer instance
    solution1_type : str
        First solution type to compare
    solution2_type : str
        Second solution type to compare
        
    Returns
    -------
    dict
        Comparison results including overlap and differences
    """
    sol1 = getattr(minimizer, f"{solution1_type}_solution")(as_strings=True)
    sol2 = getattr(minimizer, f"{solution2_type}_solution")(as_strings=True)
    
    sol1_set = set(sol1)
    sol2_set = set(sol2)
    
    return {
        f"{solution1_type}_only": list(sol1_set - sol2_set),
        f"{solution2_type}_only": list(sol2_set - sol1_set),
        "common_terms": list(sol1_set & sol2_set),
        f"{solution1_type}_count": len(sol1),
        f"{solution2_type}_count": len(sol2),
        "common_count": len(sol1_set & sol2_set)
    }