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
    """
    Generate a random vector of +1 and -1 values, one for each switch.

    Each switch is set randomly to either +1 (on) or -1 (off), with equal probability.
    This is how we create a random starting point for the network to begin searching for solutions.
    """
    return 2 * rng.integers(0, 2, size=num_switches) - 1


# ─── PROBLEM GENERATION ────────────────────────────────────

def random_sign(positive_probability: float, rng: np.random.Generator) -> float:
    """Randomly return +1.0 or -1.0, biased by the given probability.

    For example, if positive_probability is 0.8, then 80% of the time
    the result will be +1.0, and 20% of the time it will be -1.0.
    """
    return 1.0 if rng.random() < positive_probability else -1.0


def connection_strength(switch_i: int, switch_j: int, module_size: int,
                        intra_strength: float, inter_strength: float) -> float:
    """Return the connection magnitude for a pair of switches.

    The problem is divided into groups ("modules") of switches.
    Switches within the same module are strongly connected (intra_strength),
    while switches in different modules are weakly connected (inter_strength).
    This modular structure is what makes the problem hard — the network must
    discover these groups to solve it well.
    """
    same_module: bool = (switch_i // module_size) == (switch_j // module_size)
    return intra_strength if same_module else inter_strength


def generate_modular_problem(n_modules: int, module_size: int, intra_strength: float,
                             inter_strength: float, positive_bias_pct: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Build the constraint matrix ("alpha") that defines the optimisation problem.

    Each entry in the matrix describes the relationship between two switches:
      - A positive value means the two switches prefer to be in the same state (both on or both off).
      - A negative value means they prefer to be in opposite states.

    The matrix is symmetric: the relationship from switch A to B is the same as from B to A.
    A switch has no relationship with itself (the diagonal is zero).

    The 'positive_bias_pct' controls the balance: e.g. 80 means roughly 80% of
    relationships will be "prefer to agree" and 20% will be "prefer to disagree".
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
    """Score how well a given switch configuration satisfies the original constraints.

    The "energy" is a single number summarising how good the solution is.
    A more negative energy means more constraints are satisfied — so lower is better.

    Importantly, energy is always measured against the original problem (alpha),
    never against the modified weights that Hebbian learning produces.
    This ensures we are always measuring genuine solution quality.
    """
    return float(-0.5 * state @ alpha @ state)


# ─── LOCAL FIELD ────────────────────────────────────────────

def local_field(weights: np.ndarray, state: np.ndarray, node: int) -> float:
    """Calculate the total influence on a single switch from all the other switches.

    Each neighbouring switch exerts a push or pull (via the weights) on this switch.
    The local field is the sum of all those influences.  If the total is positive,
    the switch "wants" to be +1; if negative, it "wants" to be -1.
    """
    return weights[node] @ state


def sign_of(field_value: float) -> int:
    """Convert a field value to a switch state: +1 if positive or zero, -1 if negative."""
    return 1 if field_value >= 0 else -1


# ─── RELAXATION ─────────────────────────────────────────────

def update_one_node(state: np.ndarray, weights: np.ndarray,
                    rng: np.random.Generator) -> int:
    """Pick one switch at random and set it to whichever value (+1 or -1)
    its neighbours are pushing it towards.

    This is the basic building block of how the network searches for solutions:
    one switch at a time, it adjusts itself to better satisfy its local constraints.

    Returns the index of the switch that was updated.
    """
    node: int = rng.integers(len(state))
    state[node] = sign_of(local_field(weights, state, node))
    return node


def relax(state: np.ndarray, weights: np.ndarray, tau: int,
          rng: np.random.Generator) -> np.ndarray:
    """Let the network settle by updating tau randomly chosen switches in sequence.

    Each update nudges one switch towards a locally better configuration.
    After tau steps, the network has typically settled into a stable state
    (a "local minimum") — a solution that can't be improved by flipping
    any single switch.
    """
    for _ in range(tau):
        update_one_node(state, weights, rng)
    return state


# ─── HEBBIAN LEARNING ──────────────────────────────────────

def hebbian_update(weights: np.ndarray, state: np.ndarray, delta: float) -> None:
    """
    Apply one step of Hebbian learning to the weight matrix.

    For every pair of switches, if they are in the same state (both +1 or both -1),
    their connection is strengthened. If they are in opposite states, their connection is weakened.
    This is the basic idea of "Hebbian learning": switches that fire together, wire together.

    The diagonal of the matrix (self-connections) is always kept at zero, because a switch
    does not influence itself in this model.
    """
    weights += delta * np.outer(state, state)
    np.fill_diagonal(weights, 0)


def batched_hebbian_update(weights: np.ndarray, state: np.ndarray,
                          delta: float, num_steps: int) -> None:
    """
    Apply several Hebbian learning steps all at once, as if we had repeated the same update multiple times.

    This is a speed optimization: instead of updating the weights every single time the state stays the same,
    we "batch" (collect) those updates and apply them all together in one go. This is called "flushing" the batch.
    Flushing means we take all the pending (unapplied) changes and write them to the weights at once.
    """
    if num_steps > 0:
        weights += (delta * num_steps) * np.outer(state, state)
        np.fill_diagonal(weights, 0)


def effective_local_field(weights: np.ndarray, state: np.ndarray, node: int,
                         delta: float, pending_steps: int) -> float:
    """
    Compute the "local field" (the total influence) on a given switch, including any learning updates
    that have not yet been applied (not yet flushed).

    When we batch Hebbian updates, we keep track of how many updates are "pending" (not yet written to the weights).
    This function calculates what the field would be if we had already applied those updates, without actually changing the weights yet.

    The formula adds together:
        - the current influence from the weights
        - the extra influence from the pending (unapplied) Hebbian updates

    This lets the network behave as if learning is happening instantly, even though we only update the weights in batches.
    """
    num_switches: int = len(state)
    base_field: float = weights[node] @ state
    pending_contribution: float = delta * pending_steps * state[node] * (num_switches - 1)
    return base_field + pending_contribution


# ─── TRACKING BEST SOLUTION ────────────────────────────────

def track_best(energy_value: float, state: np.ndarray,
               current_best_energy: float,
               current_best_state: np.ndarray | None) -> tuple[float, np.ndarray]:
    """Keep track of the best solution found so far.

    Compares a new candidate against the current best.  If the new one
    has lower energy (i.e. is a better solution), it becomes the new best.
    Otherwise the previous best is kept.
    """
    if energy_value < current_best_energy:
        return energy_value, state.copy()
    return current_best_energy, current_best_state


# ─── BASELINE EXPERIMENT ───────────────────────────────────

def run_baseline(alpha: np.ndarray, num_relaxations: int,
                 rng: np.random.Generator,
                 tau: int | None = None) -> tuple[list[float], float, np.ndarray, list[np.ndarray]]:
    """Phase 1: repeated relaxation WITHOUT learning (the control condition).

    For each relaxation:
      1. Start from a new random switch configuration.
      2. Let the network settle for tau steps.
      3. Score the resulting configuration against the original problem.

    The weights never change — no learning takes place — so every relaxation
    is an independent attempt at solving the same fixed problem.  This gives
    us a baseline to compare the learning condition against.

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
    """Phase 2: repeated relaxation WITH concurrent Hebbian learning.

    This is the same as the baseline, except that the network also learns
    while it searches.  At every single switch update during relaxation,
    Hebbian learning adjusts the weights: switches that are in the same state
    get a stronger connection, and switches in opposite states get a weaker one.

    Over many relaxations, this gradually reshapes the problem: pairs of switches
    that frequently agree get stronger connections, which channels future
    relaxations towards better solutions.

    As a speed optimisation, when the state doesn't change between updates,
    the learning steps are collected and applied in a single batch ("flushed")
    only when the state finally does change.  The switch update takes these
    pending changes into account so the result is identical to updating every step.

    Energy is always scored against the *original* problem (alpha), not the
    modified weights, so we are measuring genuine solution quality.

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

            # Calculate the influence on this node, including any learning updates
            # that have not yet been applied to the weights (see above for explanation).
            h: float = effective_local_field(
                weights, state, node, delta, pending_steps,
            )
            new_value: int = sign_of(h)

            # If the node's state would change, we "flush" (apply) all the pending
            # Hebbian updates to the weights, then reset the pending counter.
            # This batching makes the algorithm much faster, but the logic is the same as
            # updating the weights every single time the state changes.
            if new_value != state[node]:
                batched_hebbian_update(weights, state, delta, pending_steps)
                pending_steps = 0
                state[node] = new_value

        # After finishing all updates for this relaxation, flush any remaining pending updates.
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
    """Count how many distinct solutions (energy levels) were found.

    Energies are rounded to the nearest integer before comparing,
    so very similar solutions are treated as the same.
    """
    return len(set(int(round(e)) for e in energies))


def running_minimum(values: list[float]) -> list[float]:
    """Track the best (lowest) energy seen so far at each step.

    The i-th entry is the minimum of all values from the start up to position i.
    This shows how the best-known solution improves over the course of the experiment.
    """
    return list(np.minimum.accumulate(values))


def analyse_results(energies_base: list[float], energies_learn: list[float],
                    best_e_base: float, best_e_learn: float,
                    num_relaxations: int) -> dict[str, object]:
    """Compare the baseline and learning results and produce summary statistics.

    The 'tail' refers to the final 20% of learning relaxations.  By that point,
    learning has typically settled, so tail statistics tell us how well the
    network performs once it has finished adapting — rather than including
    the early exploratory phase where it is still learning the problem structure.
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
