"""
Closed Dependency Engine (CDE) – Simulation Module
Author: Muhammed Ferit Duman
Model: The Closed Dependency Engine for Social Complexity (CDE)
Description:
    This module provides a simple simulation wrapper around the core CDE
    deterministic system (implemented in cde_core.py). It handles:
    - Iterative model updates
    - Recording system trajectories
    - Scenario-level execution
    - Exporting results for downstream analysis
"""

import json
import pandas as pd
from typing import Dict, List
from cde_core import cde_step


# -------------------------------------------------------------------
# 1. Run a single simulation trajectory
# -------------------------------------------------------------------
def run_simulation(
    A_list: List[float],
    V_list: List[float],
    SigmaT0: float,
    steps: int = 50
) -> Dict[str, List[float]]:
    """
    Runs the CDE system for a given number of steps.

    Args:
        A_list: Ontological weights (Yb, Ys, Yc)
        V_list: Transmission factors (vb, vs, vc)
        SigmaT0: Initial residue level
        steps: Number of iteration steps

    Returns:
        Dictionary of recorded trajectories.
    """

    SigmaT = SigmaT0
    history = {
        "step": [],
        "P": [],
        "S_value": [],
        "SigmaT": [],
        "Regime": [],
        "Phase": []
    }

    for t in range(steps):
        out = cde_step(A_list, V_list, SigmaT)

        history["step"].append(t)
        history["P"].append(out["P"])
        history["S_value"].append(out["S_value"])
        history["SigmaT"].append(out["SigmaT_new"])
        history["Regime"].append(out["Regime"])
        history["Phase"].append(out["Phase"])

        SigmaT = out["SigmaT_new"]

    return history


# -------------------------------------------------------------------
# 2. Convert results to a DataFrame
# -------------------------------------------------------------------
def to_dataframe(history: Dict[str, List[float]]) -> pd.DataFrame:
    return pd.DataFrame(history)


# -------------------------------------------------------------------
# 3. Export results to CSV or JSON
# -------------------------------------------------------------------
def export_results(history: Dict[str, List[float]], filename: str):
    df = pd.DataFrame(history)

    if filename.endswith(".csv"):
        df.to_csv(filename, index=False)
    elif filename.endswith(".json"):
        df.to_json(filename, orient="records", indent=2)
    else:
        raise ValueError("File extension must be .csv or .json")


# -------------------------------------------------------------------
# 4. Quick Test (runs if executed directly)
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Running quick CDE simulation test...")

    A = [1.0, 0.8, 0.6]
    V = [0.9, 1.2, 0.7]
    SigmaT0 = 0.0

    history = run_simulation(A, V, SigmaT0, steps=20)
    df = to_dataframe(history)

    print(df.head())
    export_results(history, "cde_output_test.json")
    print("Test completed. Output saved to cde_output_test.json.")

