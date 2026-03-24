"""
Self-modelling Hopfield network algorithm.

Pure computation — no UI dependency. Used by both the Streamlit demo
(hebbian_demo_4.py) and the CLI entry point below.

Based on: Watson, Buckley & Mills — "Optimisation in Self-modelling
Complex Adaptive Systems"
"""

import argparse
import json

import numpy as np


# ─── CORE FUNCTIONS ─────────────────────────────────────────

def generate_modular_problem(n_modules, module_size, intra_strength,
                             inter_strength, positive_bias_pct, rng):
    """Build a symmetric modular constraint matrix α."""
    n = n_modules * module_size
    p_positive = positive_bias_pct / 100.0
    alpha = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same_module = (i // module_size) == (j // module_size)
            magnitude = intra_strength if same_module else inter_strength
            sign = 1.0 if rng.random() < p_positive else -1.0
            alpha[i, j] = sign * magnitude
            alpha[j, i] = alpha[i, j]
    return alpha


def relax(s, W, tau, rng):
    """Asynchronous Hopfield relaxation for exactly tau state updates.

    At each step one randomly chosen node is updated:
        s_i <- sign(sum_j w_ij s_j)

    This matches Watson et al. Eq 1.
    """
    n = len(s)
    for _ in range(tau):
        i = rng.integers(n)
        h = W[i] @ s
        s[i] = 1 if h >= 0 else -1
    return s


def true_energy(s, alpha):
    """Energy under original constraints (sum over unique pairs)."""
    return float(-0.5 * s @ alpha @ s)


def run_baseline(alpha, num_relaxations, rng, tau=None):
    """Phase 1: repeated asynchronous relaxation without learning.

    Each relaxation runs for tau = 10*N asynchronous state updates
    (Watson et al. default), then the state is recorded and the
    system is reset to a new random configuration.

    Returns (energies, best_energy, best_state, all_states).
    """
    n = alpha.shape[0]
    if tau is None:
        tau = 10 * n
    W = alpha.copy()
    energies = []
    all_states = []
    best_e = float('inf')
    best_s = None
    for _ in range(num_relaxations):
        s = 2 * rng.integers(0, 2, size=n) - 1
        s = relax(s, W, tau, rng)
        e = true_energy(s, alpha)
        energies.append(e)
        all_states.append(s.copy())
        if e < best_e:
            best_e = e
            best_s = s.copy()
    return energies, best_e, best_s, all_states


def run_with_learning(alpha, num_relaxations, delta, rng, tau=None):
    """Phase 2: repeated asynchronous relaxation with concurrent Hebbian learning.

    Exactly replicates Watson et al.: at every single state update
    during relaxation, we apply Hebbian learning to all connections:
        w_ij(t+1) = w_ij(t) + delta * s_i(t) * s_j(t)   (Eq 3)

    delta is per-update (Watson: delta = 0.00025 / 10N for the
    modular problem). tau = 10*N updates per relaxation.

    Optimised: consecutive timesteps where the state doesn't change
    are batched into a single outer-product update scaled by the
    repeat count. The node update accounts for the pending accumulated
    Hebbian contribution analytically.

    Returns (energies, best_energy, best_state, all_states).
    """
    n = alpha.shape[0]
    if tau is None:
        tau = 10 * n
    W = alpha.copy()
    energies = []
    all_states = []
    best_e = float('inf')
    best_s = None
    for _ in range(num_relaxations):
        s = 2 * rng.integers(0, 2, size=n) - 1
        pending = 0  # unflushed Hebbian steps
        for _ in range(tau):
            pending += 1
            i = rng.integers(n)
            # Effective field at node i including pending Hebbian:
            # W_eff[i] = W[i] + pending*delta*(s[i]*s with diag zeroed)
            # W_eff[i] @ s = W[i]@s + pending*delta*s[i]*(s@s - 1)
            h = W[i] @ s + delta * pending * s[i] * (n - 1)
            new_val = 1 if h >= 0 else -1
            if new_val != s[i]:
                W += (delta * pending) * np.outer(s, s)
                np.fill_diagonal(W, 0)
                pending = 0
                s[i] = new_val
        if pending > 0:
            W += (delta * pending) * np.outer(s, s)
            np.fill_diagonal(W, 0)
        e = true_energy(s, alpha)
        energies.append(e)
        all_states.append(s.copy())
        if e < best_e:
            best_e = e
            best_s = s.copy()
    return energies, best_e, best_s, all_states


def run_hierarchical(alpha, n_modules, module_size, num_relaxations,
                     delta, rng, tau=None, tau_multiplier=10):
    """Phase 3: hierarchical two-level Hopfield with Hebbian learning.

    Level 1 — each group of `module_size` switches runs its own
    relaxation + Hebbian learning using only its intra-group
    connections. After `num_relaxations` rounds, each group has
    converged to a best internal state s_g.

    Level 2 — each group's output is a binary polarity: +1 (keep s_g)
    or -1 (flip to -s_g). The effective weight between groups g and h
    is W_gh = sum_{i in g, j in h} alpha_ij * s_g[i] * s_h[j], which
    captures whether the two groups "want" the same or opposite polarity.
    A 30-node Hopfield network (one super-node per group) then runs
    relaxation + Hebbian learning to find the best arrangement of
    group polarities.

    Reconstruction — for each group g, if the Level-2 state is +1 use
    s_g; if -1 use -s_g. Concatenate to get the full N-switch state.
    Energy is always measured against the original alpha.

    Returns (energies_l2, best_energy, best_state, all_states,
             group_states, meta_info).
    """
    n = n_modules * module_size
    if tau is None:
        tau = tau_multiplier * module_size  # tau relative to group size for Level 1

    # ── Level 1: solve each group independently ──────────────
    group_best_states = []  # best internal state per group
    group_energies = []     # best energy per group (internal only)
    for g in range(n_modules):
        start = g * module_size
        end = start + module_size
        alpha_g = alpha[start:end, start:end]
        tau_g = tau_multiplier * module_size
        delta_g = delta / tau_g if tau_g > 0 else delta

        # Run Hebbian learning within this group
        _, best_e_g, best_s_g, _ = run_with_learning(
            alpha_g, num_relaxations, delta_g, rng, tau=tau_g,
        )
        group_best_states.append(best_s_g)
        group_energies.append(best_e_g)

    # ── Build Level-2 effective weight matrix ────────────────
    W_meta = np.zeros((n_modules, n_modules))
    for g in range(n_modules):
        for h in range(g + 1, n_modules):
            s_g = group_best_states[g]
            s_h = group_best_states[h]
            g_start = g * module_size
            h_start = h * module_size
            # Sum of alpha_ij * s_g[i-g_start] * s_h[j-h_start]
            alpha_gh = alpha[g_start:g_start + module_size,
                             h_start:h_start + module_size]
            w = float(s_g @ alpha_gh @ s_h)
            W_meta[g, h] = w
            W_meta[h, g] = w

    # ── Level 2: solve the group-polarity problem ────────────
    tau_meta = tau_multiplier * n_modules
    delta_meta = delta / tau_meta if tau_meta > 0 else delta
    energies_l2_raw, _, best_polarity, all_polarities = run_with_learning(
        W_meta, num_relaxations, delta_meta, rng, tau=tau_meta,
    )

    # ── Reconstruct full states and compute true energies ────
    energies = []
    all_states = []
    best_e = float('inf')
    best_s = None
    for polarity in all_polarities:
        full_state = np.empty(n)
        for g in range(n_modules):
            start = g * module_size
            full_state[start:start + module_size] = (
                polarity[g] * group_best_states[g]
            )
        e = true_energy(full_state, alpha)
        energies.append(e)
        all_states.append(full_state.copy())
        if e < best_e:
            best_e = e
            best_s = full_state.copy()

    meta_info = {
        'group_energies': group_energies,
        'n_meta_nodes': n_modules,
        'tau_level1': tau_multiplier * module_size,
        'tau_level2': tau_meta,
    }
    return energies, best_e, best_s, all_states, group_best_states, meta_info


def analyse_results(energies_base, energies_learn, best_e_base,
                    best_e_learn, num_relaxations):
    """Compute summary statistics from both experiment phases."""
    tail = max(1, num_relaxations // 5)
    return {
        'base_mean': float(np.mean(energies_base)),
        'base_std': float(np.std(energies_base)),
        'learn_mean': float(np.mean(energies_learn)),
        'learn_std': float(np.std(energies_learn)),
        'tail': tail,
        'learn_tail_mean': float(np.mean(energies_learn[-tail:])),
        'learn_tail_std': float(np.std(energies_learn[-tail:])),
        'improvement': float(best_e_base - best_e_learn),
        'learning_won': bool(best_e_learn < best_e_base),
        'unique_base': len(set(int(round(e)) for e in energies_base)),
        'unique_learn_tail': len(set(int(round(e))
                                     for e in energies_learn[-tail:])),
        'running_best_base': list(np.minimum.accumulate(energies_base)),
        'running_best_learn': list(np.minimum.accumulate(energies_learn)),
    }


# ─── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
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

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    n = args.modules * args.module_size
    tau = args.tau_mult * n
    delta = args.delta if args.delta is not None else 0.00025 / tau
    print(f"Problem: {n} switches = {args.modules} modules × {args.module_size}")
    print(f"Intra: {args.intra}, Inter: {args.inter}, Bias: {args.bias}%")
    print(f"Relaxations: {args.relaxations}, τ: {tau} updates, "
          f"δ: {delta:.2e} per update")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()

    alpha = generate_modular_problem(
        args.modules, args.module_size, args.intra,
        args.inter, args.bias, rng,
    )

    print("Phase 1: baseline (no learning)...")
    energies_base, best_e_base, _, _ = run_baseline(
        alpha, args.relaxations, rng, tau=tau,
    )

    print("Phase 2: with Hebbian learning...")
    energies_learn, best_e_learn, _, _ = run_with_learning(
        alpha, args.relaxations, delta, rng, tau=tau,
    )

    stats = analyse_results(
        energies_base, energies_learn, best_e_base, best_e_learn,
        args.relaxations,
    )

    if args.json:
        output = {
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
        tail = stats['tail']
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
