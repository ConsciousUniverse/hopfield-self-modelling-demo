"""
Selective (rHN-S) and Generative (rHN-G) association algorithms.

Pure computation — no UI dependency.  Imports low-level primitives from
hopfield_algorithm.py but implements its own relaxation / learning loops.

Based on: Watson, Mills & Buckley — "Transformations in the Scale of
Behaviour and the Global Optimisation of Constraints in Adaptive
Networks" (Adaptive Behavior, 2011)

Public API:
    run_selective   — rHN-S: learned associations warp the energy surface
    run_generative  — rHN-G: learned associations produce macro-variations
"""

from __future__ import annotations

import numpy as np

from hopfield_algorithm import (
    random_binary_state,
    true_energy,
    relax,
    track_best,
)


# ─── LEARNING ──────────────────────────────────────────────

def _end_of_relaxation_learn(M: np.ndarray, state: np.ndarray,
                              delta: float) -> None:
    """Hebbian update applied once at the end of each relaxation.

    m_ij(τ+1) = γ[ m_ij(τ) + δ · s_i(τ) · s_j(τ) ]

    γ clips every weight to the range [-1, 1].
    """
    M += delta * np.outer(state, state)
    np.fill_diagonal(M, 0)
    np.clip(M, -1, 1, out=M)


# ─── ENERGY DELTA ──────────────────────────────────────────

def _energy_change(state: np.ndarray, new_state: np.ndarray,
                   alpha: np.ndarray) -> float:
    """Compute E(new_state) − E(state) using only the changed nodes.

    Exploits the sparsity of the state-change vector to avoid the
    full O(N²) quadratic form when only k << N nodes change.
    """
    d = new_state - state
    changed = np.nonzero(d)[0]
    if len(changed) == 0:
        return 0.0
    d_c = d[changed]
    h = alpha[changed] @ state
    return float(-d_c @ h - 0.5 * d_c @ alpha[np.ix_(changed, changed)] @ d_c)


# ─── rHN-S: SELECTIVE ASSOCIATIONS ─────────────────────────

def run_selective(
    alpha: np.ndarray,
    num_relaxations: int,
    delta: float,
    rng: np.random.Generator,
    tau: int | None = None,
) -> tuple[list[float], float, np.ndarray, list[np.ndarray]]:
    """rHN-S = rHN(I, Ω+M): selective associations warp the energy surface.

    Each relaxation uses single-bit-flip dynamics (same as rHN-0) but
    node updates are evaluated against a *modified* energy function
    E' = H(S, Ω + M) where M is a Hebbian associative memory learned
    from previous attractor states.  This enlarges the basins of
    frequently-visited, typically lower-energy solutions.

    Solutions are always scored against the *original* constraint matrix
    Ω so that we measure genuine solution quality.

    Returns (energies, best_energy, best_state, all_states).
    """
    N = alpha.shape[0]
    if tau is None:
        tau = 10 * N

    M = np.zeros((N, N))
    energies: list[float] = []
    all_states: list[np.ndarray] = []
    best_energy = float('inf')
    best_state: np.ndarray | None = None

    for _ in range(num_relaxations):
        state = random_binary_state(N, rng)
        state = relax(state, alpha + M, tau, rng)   # test uses Ω+M
        e = true_energy(state, alpha)               # score uses Ω
        energies.append(e)
        all_states.append(state.copy())
        best_energy, best_state = track_best(e, state, best_energy, best_state)
        _end_of_relaxation_learn(M, state, delta)

    return energies, best_energy, best_state, all_states


# ─── rHN-G: GENERATIVE ASSOCIATIONS ───────────────────────

def run_generative(
    alpha: np.ndarray,
    num_relaxations: int,
    delta: float,
    rng: np.random.Generator,
    tau: int | None = None,
) -> tuple[list[float], float, np.ndarray, list[np.ndarray]]:
    """rHN-G = rHN(I+M, Ω): generative associations produce macro-variations.

    Each relaxation uses a *modified* variation operator g(S) that
    produces correlated multi-variable state changes based on the
    learned matrix M.  When a randomly chosen node X is updated, all
    sufficiently correlated nodes are also set to values that reflect
    past co-activations.  The resulting candidate is tested against the
    *original* energy function E = H(S, Ω).

    This mechanism enables the network to search in combinations of
    learned modules rather than individual switches — a qualitative
    rescaling of the search space.

    Returns (energies, best_energy, best_state, all_states).
    """
    N = alpha.shape[0]
    if tau is None:
        tau = 10 * N

    M = np.zeros((N, N))
    energies: list[float] = []
    all_states: list[np.ndarray] = []
    best_energy = float('inf')
    best_state: np.ndarray | None = None

    for _ in range(num_relaxations):
        state = random_binary_state(N, rng)

        # Threshold base — constant within a relaxation since M is
        # only updated at the end.
        a = float(np.mean(np.abs(M)))
        if a >= 1.0:
            a = 1.0 - 1e-10

        for _ in range(tau):
            X = rng.integers(N)

            # Row of the correlation matrix C = I + M for node X.
            # c_XX = 1 (since m_XX = 0); c_Xj = m_Xj for j ≠ X.
            m_row = M[X]                        # view, not copy

            r = rng.uniform(a, 1.0)

            # Nodes whose correlation with X exceeds the threshold.
            mask = np.abs(m_row) > r
            mask[X] = True                      # X always participates

            # Build candidate state.
            new_state = state.copy()
            new_state[X] = -state[X]            # X is always flipped

            others = np.where(mask)[0]
            others = others[others != X]
            if others.size > 0:
                # s'_j = −sign(c_Xj) · s_X  (paper Eq. 7)
                new_state[others] = (
                    -np.sign(m_row[others]) * state[X]
                ).astype(int)

            # Accept only if energy does not increase (original Ω).
            if _energy_change(state, new_state, alpha) <= 0:
                state = new_state

        e = true_energy(state, alpha)
        energies.append(e)
        all_states.append(state.copy())
        best_energy, best_state = track_best(e, state, best_energy, best_state)

        _end_of_relaxation_learn(M, state, delta)

    return energies, best_energy, best_state, all_states
