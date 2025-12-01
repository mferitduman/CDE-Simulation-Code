"""
Closed Dependency Engine (CDE) – Core Mechanistic Implementation
Author: Canan Bozkurt
Model: The Closed Dependency Engine for Social Complexity (CDE)
Description:
    This module implements the core deterministic operators used in the CDE model:
    - T (Dependency Flows)
    - P (Power Formation)
    - S (Sacredness Stabilization)
    - Σ₁–Σ₄ Artificialization Layers
    - Σ_T Residue Accumulation
    - R & F System-Level Regime / Phase Transitions
"""

import math
from typing import List, Dict


# ------------------------------------------------------------
# 1. Y = (Yb, Ys, Yc) — Ontological Base
# ------------------------------------------------------------

class Ontology:
    def __init__(self, Yb: float, Ys: float, Yc: float):
        self.Yb = Yb   # biological
        self.Ys = Ys   # social
        self.Yc = Yc   # cognitive


# ------------------------------------------------------------
# 2. Dependency Architecture (T = g(A, V))
# ------------------------------------------------------------

def g(A: float, V: float) -> float:
    """
    Dependency flow operator.
    Monotonic: ∂T/∂A > 0 and ∂T/∂V > 0.
    """
    return max(0.0, A * V)


def compute_T_matrix(A_list: List[float], V_list: List[float]) -> List[List[float]]:
    """
    Computes the full T_ij matrix for all agents i → j.
    """
    T = []
    for Ai in A_list:
        row = [g(Ai, Vj) for Vj in V_list]
        T.append(row)
    return T


# ------------------------------------------------------------
# 3. Power Formation (P = h(Σ_in − Σ_out))
# ------------------------------------------------------------

def h(inflow: float, outflow: float) -> float:
    """
    Power = asymmetry of T flows.
    """
    return inflow - outflow


def compute_power(T: List[List[float]]) -> List[float]:
    """
    Computes P_j for each agent j.
    """
    N = len(T)
    P = []
    for j in range(N):
        inflow = sum(T[i][j] for i in range(N))
        outflow = sum(T[j][k] for k in range(N))
        P.append(h(inflow, outflow))
    return P


# ------------------------------------------------------------
# 4. Sacredness Stabilization (S = k / (1 + Var(P)))
# ------------------------------------------------------------

def compute_sacredness(P: List[float], k: float = 1.0) -> float:
    """
    Sacredness is stabilized by low power variance.
    """
    mean = sum(P) / len(P)
    var = sum((p - mean) ** 2 for p in P) / len(P)
    return k / (1 + var)


# ------------------------------------------------------------
# 5. Artificialization Layers (Σ₁ = m(T,P,S), …)
# ------------------------------------------------------------

def m_layer(T_total: float, P_total: float, S: float) -> float:
    """Σ₁ = m(T,P,S)"""
    return T_total + abs(P_total) + S


def d_layer(Sigma1: float) -> float:
    """Σ₂ = d(Σ₁)"""
    return math.log(1 + Sigma1)


def s_layer(Sigma2: float) -> float:
    """Σ₃ = s(Σ₂)"""
    return Sigma2 ** 1.5


def z_layer(Sigma3: float) -> float:
    """Σ₄ = z(Σ₃)"""
    return 1 / (1 + math.exp(-Sigma3))


# ------------------------------------------------------------
# 6. Residue Accumulation (Σ_T(t+1) = Σ_T(t) + Σ₄(t))
# ------------------------------------------------------------

def update_residue(SigmaT: float, Sigma4: float) -> float:
    return SigmaT + Sigma4


# ------------------------------------------------------------
# 7. Regime & Phase Functions (R = Φ(Σ_T), F = threshold(Σ_T))
# ------------------------------------------------------------

def regime_function(SigmaT: float) -> int:
    """
    Regime classifier:
    R1 = low-residue
    R2 = sacred-power
    R3 = bureaucratic-design
    R4 = hyper-artificialization
    """
    if SigmaT < 2:
        return 1
    elif SigmaT < 5:
        return 2
    elif SigmaT < 10:
        return 3
    else:
        return 4


def phase_transition(SigmaT: float) -> int:
    """
    Phase shifts based on structural thresholds.
    """
    if SigmaT < 2:
        return 0   # stable
    elif SigmaT < 5:
        return 1   # mild tension
    elif SigmaT < 10:
        return 2   # escalation
    else:
        return 3   # systemic phase shift


# ------------------------------------------------------------
# 8. Full CDE Step (one iteration)
# ------------------------------------------------------------

def cde_step(A_list: List[float], V_list: List[float], SigmaT: float):
    """
    Executes one full deterministic cycle of the Closed Dependency Engine.
    Returns all intermediate values.
    """

    # 1. Compute dependency flows
    T_matrix = compute_T_matrix(A_list, V_list)

    # 2. Power formation
    P = compute_power(T_matrix)

    # 3. Sacredness stabilization
    S_value = compute_sacredness(P)

    # 4. Artificialization layers
    T_total = sum(sum(row) for row in T_matrix)
    P_total = sum(P)

    Sigma1 = m_layer(T_total, P_total, S_value)
    Sigma2 = d_layer(Sigma1)
    Sigma3 = s_layer(Sigma2)
    Sigma4 = z_layer(Sigma3)

    # 5. Residue update
    SigmaT_new = update_residue(SigmaT, Sigma4)

    # 6. Regime & Phase
    R = regime_function(SigmaT_new)
    F = phase_transition(SigmaT_new)

    return {
        "T": T_matrix,
        "P": P,
        "S": S_value,
        "Sigma1": Sigma1,
        "Sigma2": Sigma2,
        "Sigma3": Sigma3,
        "Sigma4": Sigma4,
        "SigmaT_new": SigmaT_new,
        "Regime": R,
        "Phase": F
    }


# ------------------------------------------------------------
# 9. Quick Test
# ------------------------------------------------------------
if __name__ == "__main__":
    A = [1.0, 0.8, 0.6]
    V = [0.9, 1.2, 0.7]
    SigmaT = 0.0

    out = cde_step(A, V, SigmaT)
    print("CDE Step Output:")
    for k, v in out.items():
        print(k, ":", v)

