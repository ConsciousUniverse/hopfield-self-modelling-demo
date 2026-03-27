"""
Self-modelling Hopfield network algorithm.

Pure computation — no UI dependency. Used by both the Streamlit demo
(hebbian_demo_4.py) and the CLI entry point below.

Based on: Watson, Buckley & Mills — "Optimisation in Self-modelling
Complex Adaptive Systems" (2011)

Public API (used by the Streamlit demo):
    generate_modular_problem
    true_energy
    run_baseline
    run_with_learning
    analyse_results
"""

from __future__ import annotations

import argparse
import json

import numpy as np


# ─── RANDOM STATE ───────────────────────────────────────────

def random_binary_state(num_switches: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random state vector of +1 and -1 values.

    Each switch is independently set to +1 or -1 with equal
    probability.  This is the starting point for every relaxation.
    """
    return 2 * rng.integers(0, 2, size=num_switches) - 1


# ─── PROBLEM GENERATION ────────────────────────────────────

def random_sign(positive_probability: float, rng: np.random.Generator) -> float:
    """Return +1.0 with the given probability, else -1.0."""
    return 1.0 if rng.random() < positive_probability else -1.0


def connection_strength(switch_i: int, switch_j: int, module_size: int,
                        intra_strength: float, inter_strength: float) -> float:
    """Return the connection magnitude for a pair of switches.

    Switches in the same module get intra_strength (strong);
    switches in different modules get inter_strength (weak).
    """
    same_module: bool = (switch_i // module_size) == (switch_j // module_size)
    return intra_strength if same_module else inter_strength


def generate_modular_problem(n_modules: int, module_size: int, intra_strength: float,
                             inter_strength: float, positive_bias_pct: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Build a symmetric modular constraint matrix alpha.

    alpha[i][j] encodes the preference between switches i and j:
      positive → they prefer to agree (+1/+1 or -1/-1)
      negative → they prefer to disagree

    The matrix is symmetric (alpha[i][j] == alpha[j][i]) with a
    zero diagonal (no self-connections).
    """
    num_switches: int = n_modules * module_size
    positive_prob: float = positive_bias_pct / 100.0
    alpha: np.ndarray = np.zeros((num_switches, num_switches))

    for i in range(num_switches):
        for j in range(i + 1, num_switches):
            magnitude: float = connection_strength(
                i, j, module_size, intra_strength, inter_strength,
            )
            sign: float = random_sign(positive_prob, rng)
            alpha[i, j] = sign * magnitude
            alpha[j, i] = sign * magnitude

    return alpha


# ─── ENERGY ─────────────────────────────────────────────────

def true_energy(state: np.ndarray, alpha: np.ndarray) -> float:
    """Hopfield energy of a state scored against constraint matrix alpha.

    E = -0.5 * sum_ij alpha[i][j] * s[i] * s[j]

    More negative = more constraints satisfied = better solution.
    The energy is always computed against the *original* alpha,
    even when the working weights W have been modified by learning.
    """
    return float(-0.5 * state @ alpha @ state)


# ─── LOCAL FIELD ────────────────────────────────────────────

def local_field(weights: np.ndarray, state: np.ndarray, node: int) -> float:
    """Compute the local field at a single node.

    h_i = sum_j W[i][j] * s[j]

    This is the "pressure" on node i: if h_i >= 0 the node wants
    to be +1; if h_i < 0 it wants to be -1.
    """
    return weights[node] @ state


def sign_of(field_value: float) -> int:
    """Convert a local field value to a binary state: +1 or -1.

    Ties (h == 0) go to +1, matching the standard convention.
    """
    return 1 if field_value >= 0 else -1


# ─── RELAXATION ─────────────────────────────────────────────

def update_one_node(state: np.ndarray, weights: np.ndarray,
                    rng: np.random.Generator) -> int:
    """Pick a random node and align it with its local field.

    This is one asynchronous Hopfield update step (Watson Eq 1):
      1. Choose a node uniformly at random.
      2. Set s_i = sign(h_i).

    Returns the index of the node that was updated.
    """
    node: int = rng.integers(len(state))
    state[node] = sign_of(local_field(weights, state, node))
    return node


def relax(state: np.ndarray, weights: np.ndarray, tau: int,
          rng: np.random.Generator) -> np.ndarray:
    """Run tau asynchronous update steps (relaxation to a local minimum).

    The network settles by repeatedly aligning randomly chosen nodes
    with their local fields.  After tau steps the state is typically
    near a local energy minimum.
    """
    for _ in range(tau):
        update_one_node(state, weights, rng)
    return state


# ─── HEBBIAN LEARNING ──────────────────────────────────────

def hebbian_update(weights: np.ndarray, state: np.ndarray, delta: float) -> None:
    """Apply one step of Hebbian learning to the weight matrix.

    W[i][j] += delta * s[i] * s[j]   for all i != j   (Watson Eq 3)

    Pairs that currently agree get their connection strengthened;
    pairs that disagree get it weakened.  The diagonal stays zero
    (no self-connections in a Hopfield network).
    """
    weights += delta * np.outer(state, state)
    np.fill_diagonal(weights, 0)


def batched_hebbian_update(weights: np.ndarray, state: np.ndarray,
                          delta: float, num_steps: int) -> None:
    """Apply num_steps worth of identical Hebbian updates at once.

    Equivalent to calling hebbian_update() num_steps times with the
    same state, but done in a single matrix operation for speed.
    """
    if num_steps > 0:
        weights += (delta * num_steps) * np.outer(state, state)
        np.fill_diagonal(weights, 0)


def effective_local_field(weights: np.ndarray, state: np.ndarray, node: int,
                         delta: float, pending_steps: int) -> float:
    """Local field at a node including unflushed Hebbian contributions.

    When we batch Hebbian updates, we need to account for the
    pending (not yet applied) weight changes when deciding how to
    update a node.  The effective field is:

      h_i = W[i] . s  +  pending * delta * s[i] * (N - 1)

    The second term is the contribution from the pending outer-product
    updates (the diagonal term s[i]^2 = 1 is excluded because
    self-connections are always zero).
    """
    num_switches: int = len(state)
    base_field: float = weights[node] @ state
    pending_contribution: float = delta * pending_steps * state[node] * (num_switches - 1)
    return base_field + pending_contribution


# ─── TRACKING BEST SOLUTION ────────────────────────────────

def track_best(energy_value: float, state: np.ndarray,
               current_best_energy: float,
               current_best_state: np.ndarray | None) -> tuple[float, np.ndarray]:
    """Update the best-known solution if the new one is better.

    Returns (best_energy, best_state) — either the existing best
    or the new candidate if it has lower energy.
    """
    if energy_value < current_best_energy:
        return energy_value, state.copy()
    return current_best_energy, current_best_state


# ─── BASELINE EXPERIMENT ───────────────────────────────────

def run_baseline(alpha: np.ndarray, num_relaxations: int,
                 rng: np.random.Generator,
                 tau: int | None = None) -> tuple[list[float], float, np.ndarray, list[np.ndarray]]:
    """Phase 1: repeated relaxation without learning.

    For each relaxation:
      1. Start from a random binary state.
      2. Relax for tau steps (network settles to a local minimum).
      3. Record the energy scored against the original alpha.

    The weight matrix stays fixed at alpha throughout — there is
    no learning, so each relaxation is an independent sample from
    the network's attractor landscape.

    Returns (energies, best_energy, best_state, all_states).
    """
    num_switches: int = alpha.shape[0]
    if tau is None:
        tau = 10 * num_switches

    weights: np.ndarray = alpha.copy()
    energies: list[float] = []
    all_states: list[np.ndarray] = []
    best_energy: float = float('inf')
    best_state: np.ndarray | None = None

    for _ in range(num_relaxations):
        state: np.ndarray = random_binary_state(num_switches, rng)
        state = relax(state, weights, tau, rng)

        e: float = true_energy(state, alpha)
        energies.append(e)
        all_states.append(state.copy())
        best_energy, best_state = track_best(
            e, state, best_energy, best_state,
        )

    return energies, best_energy, best_state, all_states


# ─── LEARNING EXPERIMENT ───────────────────────────────────

def run_with_learning(alpha: np.ndarray, num_relaxations: int, delta: float,
                      rng: np.random.Generator,
                      tau: int | None = None) -> tuple[list[float], float, np.ndarray, list[np.ndarray]]:
    """Phase 2: repeated relaxation with concurrent Hebbian learning.

    At every single state update during relaxation, Hebbian learning
    is also applied (Watson Eq 3).  Over many relaxations, this
    reshapes the energy landscape: pairs of switches that frequently
    agree get stronger connections, funnelling future relaxations
    towards better solutions.

    Optimisation: consecutive timesteps where the state doesn't
    change are batched into a single outer-product update.  The
    node update accounts for the pending Hebbian contribution
    analytically via effective_local_field().

    Energy is always scored against the *original* alpha.

    Returns (energies, best_energy, best_state, all_states).
    """
    num_switches: int = alpha.shape[0]
    if tau is None:
        tau = 10 * num_switches

    weights: np.ndarray = alpha.copy()
    energies: list[float] = []
    all_states: list[np.ndarray] = []
    best_energy: float = float('inf')
    best_state: np.ndarray | None = None

    for _ in range(num_relaxations):
        state: np.ndarray = random_binary_state(num_switches, rng)
        pending_steps: int = 0

        for _ in range(tau):
            pending_steps += 1
            node: int = rng.integers(num_switches)

            # Compute field including unflushed Hebbian changes.
            h: float = effective_local_field(
                weights, state, node, delta, pending_steps,
            )
            new_value: int = sign_of(h)

            # Flush pending Hebbian updates only when the state
            # actually changes — this is the batching optimisation.
            if new_value != state[node]:
                batched_hebbian_update(weights, state, delta, pending_steps)
                pending_steps = 0
                state[node] = new_value

        # Flush any remaining pending updates after the relaxation.
        batched_hebbian_update(weights, state, delta, pending_steps)

        e: float = true_energy(state, alpha)
        energies.append(e)
        all_states.append(state.copy())
        best_energy, best_state = track_best(
            e, state, best_energy, best_state,
        )

    return energies, best_energy, best_state, all_states


# ─── ANALYSIS ──────────────────────────────────────────────

def count_unique_energy_levels(energies: list[float]) -> int:
    """Count distinct energy levels (rounded to nearest integer)."""
    return len(set(int(round(e)) for e in energies))


def running_minimum(values: list[float]) -> list[float]:
    """Running minimum: result[i] = min(values[0], ..., values[i])."""
    return list(np.minimum.accumulate(values))


def analyse_results(energies_base: list[float], energies_learn: list[float],
                    best_e_base: float, best_e_learn: float,
                    num_relaxations: int) -> dict[str, object]:
    """Compute summary statistics from both experiment phases.

    The 'tail' is the last 20% of learning relaxations — by this
    point, learning has typically converged, so tail statistics
    reflect the network's settled performance.
    """
    tail_size: int = max(1, num_relaxations // 5)
    learning_tail: list[float] = energies_learn[-tail_size:]

    return {
        'base_mean':          float(np.mean(energies_base)),
        'base_std':           float(np.std(energies_base)),
        'learn_mean':         float(np.mean(energies_learn)),
        'learn_std':          float(np.std(energies_learn)),
        'tail':               tail_size,
        'learn_tail_mean':    float(np.mean(learning_tail)),
        'learn_tail_std':     float(np.std(learning_tail)),
        'improvement':        float(best_e_base - best_e_learn),
        'learning_won':       bool(best_e_learn < best_e_base),
        'unique_base':        count_unique_energy_levels(energies_base),
        'unique_learn_tail':  count_unique_energy_levels(learning_tail),
        'running_best_base':  running_minimum(energies_base),
        'running_best_learn': running_minimum(energies_learn),
    }


# ─── CLI ────────────────────────────────────────────────────

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the Watson self-modelling Hopfield experiment.",
    )
    parser.add_argument("--modules", type=int, default=30,
                        help="Number of modules (default: 30)")
    parser.add_argument("--module-size", type=int, default=5,
                        help="Switches per module (default: 5)")
    parser.add_argument("--intra", type=float, default=1.0,
                        help="Intra-module constraint strength (default: 1.0)")
    parser.add_argument("--inter", type=float, default=0.01,
                        help="Inter-module constraint strength (default: 0.01)")
    parser.add_argument("--bias", type=int, default=80,
                        help="Positive constraint bias %%%% (default: 80)")
    parser.add_argument("--relaxations", type=int, default=300,
                        help="Relaxations per phase (default: 300)")
    parser.add_argument("--tau-mult", type=int, default=10,
                        help="Relaxation length multiplier, tau = mult*N "
                             "(default: 10, per Watson et al.)")
    parser.add_argument("--delta", type=float, default=None,
                        help="Per-update learning rate "
                             "(default: 0.00025 / (10*N), per Watson et al.)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: none)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args: argparse.Namespace = parser.parse_args()
    rng: np.random.Generator = np.random.default_rng(args.seed)

    n: int = args.modules * args.module_size
    tau: int = args.tau_mult * n
    delta: float = args.delta if args.delta is not None else 0.00025 / tau
    print(f"Problem: {n} switches = {args.modules} modules × {args.module_size}")
    print(f"Intra: {args.intra}, Inter: {args.inter}, Bias: {args.bias}%")
    print(f"Relaxations: {args.relaxations}, τ: {tau} updates, "
          f"δ: {delta:.2e} per update")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()

    alpha: np.ndarray = generate_modular_problem(
        args.modules, args.module_size, args.intra,
        args.inter, args.bias, rng,
    )

    print("Phase 1: baseline (no learning)...")
    energies_base: list[float]
    best_e_base: float
    energies_base, best_e_base, _, _ = run_baseline(
        alpha, args.relaxations, rng, tau=tau,
    )

    print("Phase 2: with Hebbian learning...")
    energies_learn: list[float]
    best_e_learn: float
    energies_learn, best_e_learn, _, _ = run_with_learning(
        alpha, args.relaxations, delta, rng, tau=tau,
    )

    stats: dict[str, object] = analyse_results(
        energies_base, energies_learn, best_e_base, best_e_learn,
        args.relaxations,
    )

    if args.json:
        output: dict[str, object] = {
            'params': {
                'modules': args.modules,
                'module_size': args.module_size,
                'intra_strength': args.intra,
                'inter_strength': args.inter,
                'positive_bias_pct': args.bias,
                'relaxations': args.relaxations,
                'tau': tau,
                'delta_per_update': delta,
                'seed': args.seed,
            },
            'best_energy_baseline': best_e_base,
            'best_energy_learning': best_e_learn,
            'energies_baseline': energies_base,
            'energies_learning': energies_learn,
            **stats,
        }
        print(json.dumps(output, indent=2))
    else:
        tail: object = stats['tail']
        print(f"{'':─<50}")
        print(f"Baseline best energy:  {best_e_base:.0f}")
        print(f"Baseline mean energy:  {stats['base_mean']:.0f} "
              f"(std: {stats['base_std']:.0f})")
        print(f"Learning best energy:  {best_e_learn:.0f}")
        print(f"Learning mean energy:  {stats['learn_mean']:.0f} "
              f"(std: {stats['learn_std']:.0f})")
        print(f"Learning tail mean (last {tail}): "
              f"{stats['learn_tail_mean']:.0f} "
              f"(std: {stats['learn_tail_std']:.0f})")
        print(f"Improvement:           {stats['improvement']:.0f}")
        print(f"Learning won:          {stats['learning_won']}")
        print(f"Unique attractors (baseline):      {stats['unique_base']}")
        print(f"Unique attractors (learning tail):  "
              f"{stats['unique_learn_tail']}")


if __name__ == "__main__":
    main()
