import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from hopfield_algorithm import (
    generate_modular_problem,
    true_energy,
    run_baseline,
    run_with_learning,
    analyse_results,
)


# ─── Watson Fig. 1 — real data version ──────────────────────

def _watson_fig1_real(energies_base, energies_learn):
    """Three-panel figure showing how learning changes the distribution
    of attractor energies, using real experiment data.

    (a) Baseline — energies without learning (the original landscape)
    (b) Early learning — first third of relaxations with Hebbian learning
    (c) Late learning — last third of relaxations with Hebbian learning

    This is the honest, data-driven version of Watson et al., 2011, Fig. 1:
    the distribution of visited attractors narrows and shifts to lower
    energy as learning simplifies the landscape.
    """
    n = len(energies_learn)
    third = max(n // 3, 1)
    early = energies_learn[:third]
    late = energies_learn[-third:]

    panels = [
        ("(a)  No learning", energies_base),
        ("(b)  Early learning", early),
        ("(c)  Late learning", late),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    col_dot = '#3372B0'
    col_kde = '#B03333'
    rng = np.random.default_rng(42)

    for ax, (title, energies) in zip(axes, panels):
        e = np.asarray(energies, dtype=float)

        # Strip chart: jitter horizontally so dots don't stack
        jitter = rng.uniform(-0.3, 0.3, size=len(e))
        ax.scatter(jitter, e, c=col_dot, s=18, alpha=0.6,
                   edgecolors='none', zorder=3)

        # KDE curve beside the dots (mirrored violin style)
        if len(e) > 2 and np.ptp(e) > 0:
            from scipy.stats import gaussian_kde
            try:
                grid = np.linspace(e.min() - np.std(e), e.max() + np.std(e), 200)
                kde = gaussian_kde(e, bw_method=0.35)
                density = kde(grid)
                # Scale density so the curve sits beside the dots
                density = density / density.max() * 0.6
                ax.fill_betweenx(grid, -density - 0.5, -0.5,
                                 color=col_kde, alpha=0.15)
                ax.plot(-density - 0.5, grid, color=col_kde, linewidth=1.2,
                        alpha=0.6)
            except np.linalg.LinAlgError:
                pass  # all energies too similar for KDE — just show dots

        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1.5, 1.0)
        ax.tick_params(labelbottom=False, bottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    axes[0].set_ylabel(r'$E^{\alpha}_0$  (true energy)', fontsize=9)

    fig.suptitle(
        'Distribution of attractor energies narrows as learning simplifies '
        'the landscape',
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    return fig


# ─── UI ─────────────────────────────────────────────────────

st.set_page_config(page_title="Self-Modelling Hopfield Network", layout="wide")
st.title("Self-Modelling: How a Network Learns to Optimise")
st.caption(
    "Demonstrating the mechanism from Watson, Buckley & Mills 2011"
    "— *Optimisation in Self-modelling Complex Adaptive Systems* "
    "([Complexity, 2011](https://doi.org/10.1002/cplx.20346))"
)

# ─── INTRODUCTION ───────────────────────────────────────────
# (rendered as a function so it always reflects the current slider values)
def _render_intro():
    _n_mod = st.session_state.get("n_modules", 30)
    _mod_sz = st.session_state.get("module_size", 5)
    _total = _n_mod * _mod_sz if _n_mod > 0 else _mod_sz
    _bias = st.session_state.get("positive_bias", 80)
    with st.expander("How it works — click to read the full explanation"):
        st.markdown(f"""
We have **{_total} binary switches**, each ON (+1) or OFF (−1). Every pair
of switches is joined by a **connection** — a single number. A **positive**
connection between switches A and B means "A and B prefer to be the same"
(both ON or both OFF). A **negative** connection means "A and B prefer to be
different" (one ON, one OFF). The **size** of the number says how strong
that preference is: +1.0 is a strong push to agree; +0.01 is a
barely-noticeable nudge.

**What is the network trying to do?** Each connection expresses one
preference between one pair of switches — and it's **symmetrical**: if the
A–B connection says "agree," that applies equally whether you're asking
"what should A do?" or "what should B do?" But with
{_total * (_total - 1) // 2:,} connections in total, they **contradict each
other**. Here's how:

When switch A is being updated, each neighbour sends a **signal** equal to
(connection strength) × (that neighbour's current state). The sign of the
signal tells A what to do:

- B is ON (+1), connection A–B is **positive** ("agree") → signal = **+1**
  → B is pushing A toward ON.
- C is ON (+1), connection A–C is **negative** ("disagree") → signal =
  **−1** → C is pushing A toward OFF.

B says "be ON." C says "be OFF." **That's the conflict** — and A has to
pick one. Multiply this across {_total - 1} connections per switch, and
there is no arrangement that makes every connection happy simultaneously.
The challenge is to find the arrangement that **satisfies as many
preferences as possible, weighted by their strengths**.

We measure this with a number called the **energy**. For each connection,
if the two switches are doing what that connection wants (e.g. both ON when
the connection is positive), that connection is *satisfied* and contributes
a negative amount to the energy — pulling it down. If they're doing the
opposite, that connection is *frustrated* and pushes the energy up. The
total energy is the sum across all {_total * (_total - 1) // 2:,}
connections. So **lower energy = more preferences satisfied overall**. The
network's job is to **minimise energy** — which means finding the best
compromise across all those conflicting preferences.

This is *not* pattern recall (there is no stored target to recover). Nobody
knows what the best arrangement looks like in advance — it is a
combinatorial optimisation puzzle over 2^{_total} possible arrangements.

The switches are arranged in **{_n_mod} groups of {_mod_sz}**. Every pair of
switches still has its own individual connection — **including pairs in
different groups**. The only difference is the strength:

- **Within a group** (e.g. switches 1-2, 1-3, 2-3 inside the same group of
  {_mod_sz}): each connection has magnitude
  **{st.session_state.get('intra_strength', 1.0)}** — a strong pull.
- **Between groups** (e.g. switch 2 in group A ↔ switch 14 in group C):
  each connection has magnitude
  **{st.session_state.get('inter_strength', 0.01)}** — a very weak pull.

There is **no group-level computation** — groups are not treated as
single nodes. Each of the {_total} switches talks to every other switch
individually. With {_mod_sz} switches per group, there are
{_mod_sz * (_mod_sz - 1) // 2} connections inside each group, and each
switch also has its own weak connection to each of the {_total - _mod_sz}
switches in other groups. That's {_total - _mod_sz} weak connections per
switch — so the weak inter-group connections vastly outnumber the strong
intra-group ones, and their collective effect matters.

**Why groups?** This is the key to the whole demonstration. Within a group,
the strong connections are easy for switches to satisfy locally — every
group quickly finds a stable internal arrangement. But the many weak
connections *between* groups are hard to coordinate, because no individual
switch feels much pressure from any one of them.

**The {_bias}% positive bias:** When the connections are first generated
(once, before any phase begins), each one is randomly assigned its sign:
{_bias}% are made positive ("agree") and {100 - _bias}% are made negative
("disagree"). The **magnitudes** are set by the strength sliders (strong
within a group, weak between groups). Once generated, these connections
define the puzzle that all three phases will work on.

This bias isn't about making the problem easy — it's about creating
*structure*. Because most connections within a group say "agree," switches
in the same group tend to end up the same way. But there's enough
randomness that different starting arrangements settle into *different*
locally-stable patterns.

This demo runs the system in **three phases**, all working on the same
set of connections:

1. **Without learning (baseline)** — the network tries many random starting
   arrangements. Here is exactly what happens inside one "relaxation":

   We start with all {_total} switches set to random ON/OFF values.
   Then we repeat the following single action
   **{st.session_state.get('tau_multiplier', 10) * _total:,} times in a row**
   ({st.session_state.get('tau_multiplier', 10)} × {_total} switches):

   **Pick one switch at random.** Look at the current state of every
   other switch, multiply each by its connection to the chosen switch,
   and add them all up. If the total is positive, set the chosen switch
   to ON (+1). If negative, set it to OFF (−1). **Only this one switch
   can change; everything else stays put.**

   That's the entire action — choose one switch, update it, done. Then
   immediately do it again: pick another switch at random (it could be
   the same one), update it using the **current** states of all the
   others — including any switch that just changed a moment ago. Each
   update sees the result of every update before it, so changes cascade:
   switch 42 flips, which later causes switch 7 to flip (because 7 has
   a strong connection to 42), which later causes switch 91 to respond,
   and so on. Over
   {st.session_state.get('tau_multiplier', 10) * _total:,} of these
   one-at-a-time updates, the system **settles towards a compromise**
   where most switches have stopped flipping.

   **Energy is measured once, at the very end** — after all
   {st.session_state.get('tau_multiplier', 10) * _total:,} updates, we
   freeze the switches where they are and calculate the total energy
   (summing over all {_total * (_total - 1) // 2:,} connections: each
   satisfied preference subtracts from the energy, each frustrated one
   adds). This single number is the score for this relaxation.

   We record that energy, then **throw away the switch arrangement
   entirely** and start fresh from a new random one. The **connections
   never change** throughout this phase — every connection keeps its
   original sign (agree or disagree) and its original magnitude, exactly
   as generated. Nothing at all carries from one relaxation to the next.
   Each attempt is completely independent, like rolling dice again.

   But here's what's interesting: even though each relaxation starts from
   a different random arrangement, the **same connections** are pulling on
   the switches every time. If the connection between switches 7 and 23
   is strong and positive, it pushes them to agree in *every* relaxation —
   sometimes both ON, sometimes both OFF, but they tend to match. A switch
   connected to many neighbours that disagree with each other — some
   pushing it ON, others pushing it OFF — won't settle as consistently. So
   across hundreds of independent relaxations, **certain patterns keep
   recurring** — not because anything is being remembered, but because
   the same fixed connections keep producing similar outcomes. These
   recurring regularities are exactly what Phase 2 will learn to exploit.

   Because the weak inter-group connections create lots of these
   tug-of-war situations, the network usually gets *stuck* in a mediocre local
   arrangement — not terrible, but far from the best possible.

2. **Flat Hebbian learning** — the same switch-by-switch process, but now
   the connection strengths **are not fixed**. At every single state update
   (not just at the end), we make a **tiny** adjustment to **every
   connection in the whole network** — not just the connections of the
   switch that was just updated. We look at the current state of *all*
   switches: every pair that are currently the same get a slightly
   stronger "agree" connection; every pair that are currently different
   get a slightly stronger "disagree" connection. Between relaxations, **the switch
   arrangement is still thrown away and randomised** — but the adjusted
   connection strengths are kept. So the connections are the network's
   memory: they carry forward what was learned from every previous
   relaxation. One nudge is almost invisible. But over hundreds of
   relaxations, the nudges **accumulate** — and they accumulate fastest
   for the recurring patterns that appear across many different settled
   arrangements. This gradually reshapes the energy landscape: mediocre
   solutions get shallower, and good solutions get deeper. Eventually
   the network reliably finds solutions **better than any it has seen
   before** — it has *generalised* from many mediocre experiences into
   something superior.

To see what happens **without** modular structure, set the number of
modules to **0**. This creates an unstructured problem with uniform
connection strengths — every pair of switches is connected equally
strongly. Because there are no recurring sub-patterns for learning
to latch onto, the improvement is typically much smaller.
""")


def _render_info_sections():
    """Render intro + glossary + math expanders (called after results)."""
    _render_intro()
    _render_glossary()
    _render_math()


# ─── PLAIN-ENGLISH GLOSSARY ─────────────────────────────────
def _render_glossary():
    _n_mod = st.session_state.get('n_modules', _WATSON_DEFAULTS['n_modules'])
    _mod_sz = st.session_state.get('module_size', _WATSON_DEFAULTS['module_size'])
    _N = _n_mod * _mod_sz if _n_mod > 0 else _mod_sz
    _tau_m = st.session_state.get('tau_multiplier', _WATSON_DEFAULTS['tau_multiplier'])
    _tau = _tau_m * _N
    with st.expander("Glossary — what the jargon means"):
        st.markdown(f"""
| Term | What it means |
|------|---------------|
| **Constraint** | A rule between two switches, e.g. "switch 3 and switch 7 should be the same." Rules within a group are strong; rules between groups are weak but collectively significant. |
| **Energy** | A single number measuring how badly the current arrangement violates the constraints. Lower energy = fewer violations = better solution. |
| **Relaxation** | One "attempt" at solving the problem. The network starts from a random arrangement, then updates **one randomly-chosen switch at a time** for exactly **τ = {_tau:,}** steps (relaxation length multiplier ({_tau_m}) × N ({_N}) switches). By the end, the network has usually settled near a local minimum. |
| **Local minimum** | A settled arrangement where no *individual* switch can improve things by flipping. It's stable, but it might not be the best overall — just the nearest stable point from where you started. |
| **Global minimum** | The best possible arrangement across *all* configurations — the one that satisfies the most constraints. With 150 switches this is extremely hard to find. |
| **Basin of attraction** | The set of all random starting points that lead to the same settled arrangement. Bigger basin = more starting points lead there = this arrangement is found more often. |
| **Hebbian learning** | "What fires together, wires together." Applied **concurrently at every state update** during relaxation — not just at the end. Each tiny nudge strengthens the connection between switches that are currently in the same state. |
| **Attractor** | Another word for a local minimum — a stable arrangement that the system is "attracted" toward during relaxation. |
| **Hamming space** | A way of thinking about distance between switch arrangements. Two arrangements are "close" if they differ in only a few switches, and "far apart" if they differ in many. The **Hamming distance** between two arrangements is simply the number of switches that are different. For example, if arrangement A and arrangement B disagree on 12 out of 150 switches, they are 12 apart in Hamming space. |
| **Weighted Max-2-SAT** | The formal name for the type of problem the network is solving. Each pairwise constraint involves exactly **2** variables (switches) and has a weight (|α|). The goal is to **maximise** the total weight of satisfied constraints — equivalently, to **minimise** the energy. The modular structure means intra-group clauses carry large weights and inter-group clauses carry small weights. |
""")


# ─── DETAILED MATH WITH EXAMPLES ───────────────────────────
def _render_math():
    with st.expander("The constraint problem (with example)"):
        st.markdown(r"""
A **constraint** is just a rule between two switches. For example:
"switch 1 and switch 2 should be the same" or "switch 1 and switch 3
should be different." Every pair of switches has exactly one such rule,
assigned randomly. With 100 switches, that gives us ~5,000 rules.
The challenge: satisfy as many of these rules as possible *at the same
time*.

Mathematically, each constraint is a number $\alpha_{ij}$:
Each pair of switches $(i, j)$ has a constraint $\alpha_{ij}$:

| $\alpha_{ij}$ | Meaning |
|:-:|---------|
| $+1$ | switches $i$ and $j$ should be the **same** |
| $-1$ | switches $i$ and $j$ should be **different** |

**Example -- 3 switches:**

| Pair | $\alpha$ | Rule |
|------|:--------:|------|
| 1-2  | $+1$     | same |
| 1-3  | $-1$     | different |
| 2-3  | $-1$     | different |

Try $s = [+1,\; +1,\; -1]$:

- Switches 1 and 2 are the same -- constraint says "same" -- **satisfied**
- Switches 1 and 3 are different -- constraint says "different" -- **satisfied**
- Switches 2 and 3 are different -- constraint says "different" -- **satisfied**

All 3 constraints satisfied! But with 100 switches and ~5,000 random
constraints, satisfying everything is usually impossible. We want to
satisfy **as many as we can**.
""")

    with st.expander("Energy -- measuring solution quality"):
        st.markdown(r"""
We need a single number that tells us "how good is this arrangement?"
That number is called **energy**. It adds up the result of every
constraint: satisfied constraints push the number down, violated ones push
it up. So **lower energy = better solution**. The worst possible energy
would mean every constraint is violated; the best would mean every
constraint is satisfied.

The formula:

$$E = -\sum_{i < j} \alpha_{ij} \; s_i \; s_j$$

Each term $\alpha_{ij} \, s_i \, s_j$ equals:

- $+1$ when the constraint is **satisfied** (pushes energy *down*)
- $-1$ when it's **violated** (pushes energy *up*)

**Example:** $s = [+1,\; +1,\; -1]$ with the constraints above:

| Pair | $\alpha_{ij}$ | $s_i \times s_j$ | Product | Satisfied? |
|------|:-:|:-:|:-:|:-:|
| 1-2  | $+1$ | $(+1)(+1) = +1$ | $+1$ | Yes |
| 1-3  | $-1$ | $(+1)(-1) = -1$ | $+1$ | Yes |
| 2-3  | $-1$ | $(+1)(-1) = -1$ | $+1$ | Yes |

$E = -(1 + 1 + 1) = -3$ -- the lowest possible for this problem.

A bad state $s = [+1,\; -1,\; +1]$:

| Pair | $\alpha_{ij}$ | $s_i \times s_j$ | Product |
|------|:-:|:-:|:-:|
| 1-2  | $+1$ | $(+1)(-1) = -1$ | $-1$ |
| 1-3  | $-1$ | $(+1)(+1) = +1$ | $-1$ |
| 2-3  | $-1$ | $(-1)(+1) = -1$ | $+1$ |

$E = -(-1 -1 +1) = +1$ -- higher energy, worse solution.
""")

    with st.expander("Relaxation — letting the system settle"):
        # Insert dynamic values safely without requiring f-strings inside LaTeX
        _tau_m = st.session_state.get('tau_multiplier', 10)
        _example_N = 150
        st.markdown(r"""
**Relaxation** means letting the system settle on its own, with no
outside intervention. At each step, **one switch is picked at random** and
updated: it looks at all its neighbours' current states and the connection
strengths, and flips (or stays) accordingly. This repeats for exactly
$\tau = """ + f"{_tau_m}" + r"""N$ steps — e.g. """ + f"{_tau_m * _example_N:,}" + r""" steps for """ + f"{_example_N}" + r""" switches.

The update rule (Watson et al., 2011, Eq. 1):

$$s_i \leftarrow \operatorname{sign}\!\left(\sum_j w_{ij}\, s_j\right)$$

In words: "pick a switch at random. Add up all the signals from other
switches (each weighted by the connection strength). If the total is
positive, set it to $+1$. If negative, set it to $-1$."

This is **asynchronous** update — only one switch changes at a time, using
the most up-to-date values of all its neighbours.

**Example:** Starting from $s = [+1,\; -1,\; +1]$ (the bad state):

$$w_{12}\, s_2 + w_{13}\, s_3 = (+1)(-1) + (-1)(+1) = -2$$

Since $-2 < 0$, switch 1 flips to $-1$. The state is now $[-1,\; -1,\; +1]$.

Next, another switch is picked at random. Eventually, after many such
steps, the network has settled near a local minimum — a stable arrangement
where individual switches have little incentive to flip.

**The problem:** from different random starts, the network falls into
**different** local minima. With 150 switches, there are many such minima,
and most are mediocre. The global optimum is just one among many, and
plain relaxation rarely finds it.
""")

    with st.expander("Hebbian learning -- tiny nudges that accumulate"):
        st.markdown(r"""
Once the system has settled, we look at the result and make a **tiny**
adjustment to the connection strengths. The rule is simple: if two switches
ended up in the same state (both ON or both OFF), strengthen the
"agree" connection between them very slightly. If they ended up in
different states (one ON, one OFF), strengthen the "disagree" connection
slightly. One adjustment is almost invisible. But after hundreds of
trials, the adjustments **accumulate** — especially for pairs of switches
that *consistently* end up the same way across many different settled
arrangements.

Crucially, this only works **in the limit of low learning rates** — the
learning rate $\delta$ must be small enough that the system visits a
sufficient sample of attractors between weight updates. That way the
Hebbian nudges accumulate only the most reliable, recurring sub-patterns
rather than overfitting to transient fluctuations.

The formula:

$$\Delta w_{ij} = \delta \cdot s_i \cdot s_j \qquad (\delta \text{ is very small})$$

This slightly **strengthens** connections between switches that ended up in
the same state, and slightly **weakens** connections between switches that
ended up different.

**Example:** Network settled to $s = [-1,\; -1,\; +1]$:

| Pair | $s_i \cdot s_j$ | $\Delta w_{ij}$ | Effect |
|------|:-:|:-:|---------|
| 1-2  | $(-1)(-1) = +1$ | $+\delta$ | Slightly more "want same" |
| 1-3  | $(-1)(+1) = -1$ | $-\delta$ | Slightly more "want different" |
| 2-3  | $(-1)(+1) = -1$ | $-\delta$ | Slightly more "want different" |

Individually these nudges are tiny. But they **accumulate** over hundreds
of relaxations, and they accumulate **faster** for switch-pairs whose
relationship is *consistent* across many different local minima.
""")

    with st.expander("The key insight — why better solutions win"):
        st.markdown(r"""
Remember: a **basin of attraction** is the set of starting points that
lead to a particular settled arrangement. A bigger basin means more
random starts end up there. Think of it as how "easy to find" a
particular solution is.

Two facts combine:

**Fact 1:** Better solutions (lower energy) tend to have **bigger basins**.
So better solutions are found more often by pure chance.

**Fact 2:** Solutions found more often accumulate more Hebbian nudges,
which makes their basins **even bigger**.

Together this creates a positive feedback loop:

> Better solutions $\rightarrow$ visited more often $\rightarrow$ reinforced
> more $\rightarrow$ basins grow $\rightarrow$ visited even more
> $\rightarrow$ reinforced even more $\rightarrow$ ... $\rightarrow$ one
> attractor dominates

Because of Fact 1, the winner tends to be a **near-optimal** solution.

**The remarkable part:** the network also finds solutions it has **never
visited before**. Across many mediocre local minima, certain sub-patterns
recur ("switches 7 and 23 usually agree"). The Hebbian learning
accumulates these correlations and combines them into *new* attractors
representing superior solutions. The network **generalises** from
experience rather than simply memorising past results.
""")


# ─── USER CONTROLS ───────────────────────────────────────
# ─── WATSON DEFAULTS ────────────────────────────────────────
_WATSON_DEFAULTS = {
    "n_modules": 30,
    "module_size": 5,
    "intra_strength": 1.0,
    "inter_strength": 0.01,
    "positive_bias": 80,
    "num_relaxations": 300,
    "tau_multiplier": 10,
    "delta": 0.00025,
    "num_trials": 100,
}

# Initialise defaults once (before any widget is created)
for _k, _v in _WATSON_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Handle reset: apply defaults before widgets are instantiated
if st.session_state.get('_reset_defaults'):
    for _k, _v in _WATSON_DEFAULTS.items():
        st.session_state[_k] = _v
    st.session_state['_reset_defaults'] = False

with st.sidebar:
    st.header("Controls")

    st.subheader("Problem structure")
    _switches_slot = st.container()
    N_MODULES = st.number_input(
        "Number of modules",
        min_value=0,
        max_value=500,
        step=1,
        key="n_modules",
        help=(
            "Default **30**. "
            "More modules means more inter-group coordination is needed, "
            "making the problem harder for local search but giving "
            "learning more structure to exploit. "
            "Set to **0** for an unstructured problem (uniform connection "
            "strengths, no groups) \u2014 this shows that modular structure "
            "is what makes Hebbian learning effective."
        ),
    )
    _switches_label = (
        "Number of switches (N)" if N_MODULES == 0
        else "Switches per module"
    )
    with _switches_slot:
        MODULE_SIZE = st.number_input(
            _switches_label,
            min_value=2,
            max_value=5000,
            step=1,
            key="module_size",
            help=(
                "Default **5**. "
                "Larger modules create more internal structure within each "
                "group. The total number of switches (modules \u00d7 size) "
                "also affects how long each run takes."
            ),
        )
    INTRA_STRENGTH = st.number_input(
        "Intra-module constraint strength",
        min_value=0.1,
        max_value=5.0,
        step=0.1,
        format="%.1f",
        key="intra_strength",
        help=(
            "Default **1.0**. "
            "Stronger intra-module constraints mean each group's "
            "internal arrangement is resolved quickly and reliably "
            "during settling."
        ),
    )
    INTER_STRENGTH = st.number_input(
        "Inter-module constraint strength",
        min_value=0.001,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="inter_strength",
        help=(
            "Default **0.01**. "
            "Weak inter-group constraints are individually negligible but "
            "collectively significant \u2014 this is what makes the problem "
            "hard for local search and interesting for learning."
        ),
    )
    POSITIVE_BIAS = st.number_input(
        "Positive constraint bias (%)",
        min_value=50,
        max_value=100,
        step=5,
        key="positive_bias",
        help=(
            "Default **80%**. "
            "A bias toward positive (\"agree\") connections creates "
            "consistency in the problem, increasing the likelihood "
            "that local optima share common sub-patterns. "
            "At 50% there is no bias and learning has little to "
            "work with."
        ),
    )

    st.subheader("Experiment")
    TAU_MULT = st.number_input(
        "Relaxation length multiplier",
        min_value=1,
        max_value=500,
        step=1,
        key="tau_multiplier",
        help=(
            "Default **10**. "
            "Each relaxation runs for this number × N steps. "
            "Higher values give the network more time to settle "
            "each attempt, but each relaxation takes longer."
        ),
    )
    NUM_RELAXATIONS = st.number_input(
        "Relaxations per phase",
        min_value=100,
        max_value=50000,
        step=50,
        key="num_relaxations",
        help=(
            "Default **300**. "
            "Both phases run this many times, so the comparison "
            "is fair. More relaxations give the system more chances "
            "to explore, but take longer to run."
        ),
    )
    DELTA = st.number_input(
        "Learning rate (\u03b4)",
        min_value=0.00005,
        max_value=0.005,
        step=0.00005,
        format="%.5f",
        key="delta",
        help=(
            "Default **0.00025**. This is the *numerator*; the actual "
            "per-update rate is \u03b4 / (\u03c4). "
            "A tiny value means the network accumulates experience slowly "
            "and samples many different solutions before committing \u2014 "
            "this generally gives the best results. A larger value "
            "risks locking onto a mediocre early solution."
        ),
    )

    st.subheader("Multi-trial analysis")
    NUM_TRIALS = st.number_input(
        "Number of trials",
        min_value=1,
        max_value=500,
        step=1,
        key="num_trials",
        help=(
            "Default **100** (Watson et al., 2011). Each trial generates a new random "
            "problem instance and runs both a baseline and a learning "
            "experiment using the **same** random initial conditions and "
            "update order. This lets us measure how reliably learning "
            "outperforms the baseline across many different problems."
        ),
    )

    def _request_single():
        st.session_state.pop('results', None)
        st.session_state.pop('multi_trial_results', None)
        st.session_state.pop('_held_alpha', None)
        st.session_state.pop('_held_params', None)

    def _request_multi():
        st.session_state.pop('results', None)
        st.session_state.pop('multi_trial_results', None)
        st.session_state.pop('_held_alpha', None)
        st.session_state.pop('_held_params', None)

    def _request_rerun_same():
        # Keep alpha from previous results; clear results so page renders clean
        prev = st.session_state.get('results')
        if prev is not None and 'alpha' in prev:
            st.session_state['_held_alpha'] = prev['alpha']
            st.session_state['_held_params'] = {
                k: prev[k] for k in (
                    'N', 'n_modules', 'module_size', 'intra_strength',
                    'inter_strength', 'positive_bias', 'n_intra', 'n_inter',
                )
            }
        st.session_state.pop('results', None)
        st.session_state.pop('multi_trial_results', None)

    st.divider()
    _run_btn = st.button(
        "Run single experiment",
        use_container_width=True,
        on_click=_request_single,
    )

    # Show "Re-run same problem" only when we have a stored alpha
    _prev = st.session_state.get('results')
    _has_alpha = (
        (isinstance(_prev, dict) and 'alpha' in _prev)
        or '_held_alpha' in st.session_state
    )
    _rerun_btn = st.button(
        "Re-run same problem",
        use_container_width=True,
        on_click=_request_rerun_same,
        disabled=not _has_alpha,
        help=(
            "Run baseline + learning again on the **same** constraint "
            "matrix from the last experiment. The relaxation "
            "length (τ), learning rate (δ) and number of relaxations "
            "are re-read from the sliders; the problem structure "
            "stays fixed. Each relaxation still starts from a fresh "
            "random state, so results will vary — this reveals how "
            "much of the run-to-run difference is due to the random "
            "search vs. the problem itself."
        ),
    )

    _multi_btn = st.button(
        "Run multi-trial analysis",
        use_container_width=True,
        on_click=_request_multi,
    )
    st.button(
        "Reset to Watson defaults",
        use_container_width=True,
        on_click=lambda: st.session_state.update({'_reset_defaults': True}),
    )

RNG = np.random.default_rng()

# ─── UI HELPERS ─────────────────────────────────────────────
def state_to_img(state, n, px=10):
    n_side = int(np.ceil(np.sqrt(n)))
    n_side_y = int(np.ceil(n / n_side))
    pad = n_side * n_side_y - n
    padded = np.concatenate([state, np.zeros(pad)]) if pad > 0 else state
    grid = ((padded.reshape((n_side_y, n_side)) + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(grid)
    img = img.convert('L')
    return img.resize((n_side * px, n_side_y * px), resample=Image.NEAREST)

# ─── RUN ────────────────────────────────────────────────────
def _run_experiment(run, same_problem=False):
  if run:
    num_relaxations = st.session_state.num_relaxations
    delta = st.session_state.delta
    tau_multiplier = st.session_state.tau_multiplier

    if same_problem and '_held_alpha' in st.session_state:
        # Re-use the constraint matrix from the previous run
        alpha = st.session_state.pop('_held_alpha')
        params = st.session_state.pop('_held_params', {})
        N = params.get('N', alpha.shape[0])
        n_modules = params.get('n_modules', 0)
        module_size = params.get('module_size', N)
        intra_strength = params.get('intra_strength', 0)
        inter_strength = params.get('inter_strength', 0)
        positive_bias = params.get('positive_bias', 0)
        n_intra = params.get('n_intra', 0)
        n_inter = params.get('n_inter', 0)
    else:
      same_problem = False
      # Read current slider values from session state (always fresh,
      # even during a fragment-only re-run).
      n_modules = st.session_state.n_modules
      module_size = st.session_state.module_size
      intra_strength = st.session_state.intra_strength
      inter_strength = st.session_state.inter_strength
      positive_bias = st.session_state.positive_bias

      with st.spinner("Generating problem..."):
        if n_modules == 0:
            # Unstructured: use module_size as total N, uniform strength
            N = module_size
            uniform_strength = (intra_strength + inter_strength) / 2
            alpha = generate_modular_problem(
                1, N, uniform_strength, uniform_strength,
                positive_bias, RNG,
            )
            n_intra = 0
            n_inter = N * (N - 1) // 2
        else:
            N = n_modules * module_size
            alpha = generate_modular_problem(
                n_modules, module_size, intra_strength,
                inter_strength, positive_bias, RNG,
            )
            n_intra = n_modules * (module_size * (module_size - 1) // 2)
            n_inter = N * (N - 1) // 2 - n_intra

    N_PAIRS = N * (N - 1) // 2
    TAU = tau_multiplier * N
    DELTA_PER_UPDATE = delta / TAU

    progress = st.progress(0, text="Running baseline...")
    energies_base, best_e_base, best_s_base, _states_base = run_baseline(
        alpha, num_relaxations, RNG, tau=TAU,
    )
    progress.progress(50, text="Running with learning...")
    energies_learn, best_e_learn, best_s_learn, _states_learn = run_with_learning(
        alpha, num_relaxations, DELTA_PER_UPDATE, RNG, tau=TAU,
    )
    progress.empty()

    # Bump counter so the expander resets to expanded=True.
    st.session_state._results_run = st.session_state.get('_results_run', 0) + 1
    st.session_state._results_just_ran = True

    # Store everything in session state so results survive reruns.
    st.session_state.results = {
        'N': N,
        'n_modules': n_modules,
        'module_size': module_size,
        'intra_strength': intra_strength,
        'inter_strength': inter_strength,
        'positive_bias': positive_bias,
        'n_intra': n_intra,
        'n_inter': n_inter,
        'num_relaxations': num_relaxations,
        'tau_multiplier': tau_multiplier,
        'delta': delta,
        'energies_base': energies_base,
        'energies_learn': energies_learn,
        'best_e_base': best_e_base,
        'best_e_learn': best_e_learn,
        'best_s_base': best_s_base,
        'best_s_learn': best_s_learn,
        'alpha': alpha,
        'same_problem': same_problem,
    }


def _render_results():
  """Render results from session state (survives widget interactions)."""
  if 'results' not in st.session_state:
    return

  _run_id = st.session_state.get('_results_run', 0)
  _results_expander = st.expander(f"Single experiment results (run {_run_id})", expanded=True)
  with _results_expander:
    r = st.session_state.results
    N = r['N']
    num_relaxations = r['num_relaxations']
    energies_base = r['energies_base']
    energies_learn = r['energies_learn']
    best_e_base = r['best_e_base']
    best_e_learn = r['best_e_learn']
    best_s_base = r['best_s_base']
    best_s_learn = r['best_s_learn']
  
    if r['n_modules'] == 0:
        st.info(
            f"Generated an **unstructured** constraint problem: "
            f"**{N}** switches with **uniform** connection strength "
            f"(|\u03b1|={(r['intra_strength'] + r['inter_strength']) / 2:.3f}). "
            f"{r['positive_bias']}% of constraints are positive (\"agree\"). "
            f"There are no groups \u2014 every pair of switches is connected "
            f"equally strongly. Without modular structure there are fewer "
            f"recurring sub-patterns for learning to exploit."
        )
    else:
        st.info(
            f"Generated a **modular** constraint problem: "
            f"**{N}** switches in **{r['n_modules']}** modules of {r['module_size']}. "
            f"**{r['n_intra']}** strong intra-module constraints "
            f"(|\u03b1|={r['intra_strength']:.1f}), "
            f"**{r['n_inter']}** weak inter-module constraints "
            f"(|\u03b1|={r['inter_strength']:.3f}). "
            f"{r['positive_bias']}% of constraints are positive (\"agree\"). "
            f"This structure creates many local minima that share recurring "
            f"sub-patterns \u2014 exactly the condition where Hebbian learning "
            f"can generalise."
        )
  
    stats = analyse_results(
        energies_base, energies_learn, best_e_base, best_e_learn,
        num_relaxations,
    )
    base_mean = stats['base_mean']
    base_std = stats['base_std']
    learn_mean = stats['learn_mean']
    learn_std = stats['learn_std']
    tail = stats['tail']
    learn_tail_mean = stats['learn_tail_mean']
    learn_tail_std = stats['learn_tail_std']
    improvement = stats['improvement']
    learning_won = stats['learning_won']
    unique_base = stats['unique_base']
    unique_learn_tail = stats['unique_learn_tail']
  
    # ── Modular problem metrics ──
    st.markdown("##### Results")
    _mean_base = float(np.mean(energies_base))
    _mean_learn = float(np.mean(energies_learn))
    _mean_improvement_pct = (_mean_base - _mean_learn) / abs(_mean_base) * 100 if _mean_base != 0 else 0
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Baseline (no learning)", f"{best_e_base:.0f}")
    with m2:
        delta_e = best_e_learn - best_e_base
        st.metric(
            "With Hebbian learning",
            f"{best_e_learn:.0f}",
            delta=f"{delta_e:.0f} vs baseline",
            delta_color="inverse",
        )
    with m3:
        improvement_pct = (best_e_base - best_e_learn) / abs(best_e_base) * 100 if best_e_base != 0 else 0
        st.metric("Best-of-N improvement", f"{improvement_pct:+.1f}%")
    with m4:
        st.metric("Mean improvement", f"{_mean_improvement_pct:+.1f}%")
  
    st.markdown(
        "Each image below shows the **best state vector** the network "
        "found, laid out as a grid. Every square is one switch: "
        "**white = ON (+1)**, **black = OFF (\u22121)**, "
        "**grey = padding** (ignored \u2014 just fills the grid to a rectangle). "
        "A good solution tends to show large uniform blocks of white or "
        "black, reflecting agreement within modules."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.write("**No learning**")
        st.image(state_to_img(best_s_base, N), width=120)
        st.caption(f"Energy: {best_e_base:.0f}")
    with c2:
        st.write("**With learning**")
        st.image(state_to_img(best_s_learn, N), width=120)
        st.caption(f"Energy: {best_e_learn:.0f}")
  
    st.markdown(f"""
  **Phase 1 (no learning):** {num_relaxations} relaxations from random starts.
  
  - Attractor energies ranged from **{min(energies_base):.0f}** to **{max(energies_base):.0f}**
  - Mean energy: **{base_mean:.0f}** (std: {base_std:.0f})
  - Best energy found: **{best_e_base:.0f}**
  - Distinct energy levels visited: ~{unique_base}
  
  The network fell into many different local minima, scattered across a wide
  band. No single attractor dominated.
  
  ---
  
  **Phase 2 (with Hebbian learning):** {num_relaxations} relaxations, same random
  starting procedure, but with **concurrent Hebbian learning at every state
  update** throughout each relaxation (Watson et al., 2011, Eq. 3).
  
  - Best energy found: **{best_e_learn:.0f}**
  - Mean energy over all relaxations: **{learn_mean:.0f}** (std: {learn_std:.0f})
  - Mean energy over last {tail} relaxations: **{learn_tail_mean:.0f}** (std: {learn_tail_std:.0f})
  - Distinct energy levels in last {tail} relaxations: ~{unique_learn_tail}
  """)
  
    # Determine winner
    if best_e_learn >= best_e_base:
        st.markdown(f"""
  **Learning did not beat the baseline** this time
  (baseline best: **{best_e_base:.0f}**, with learning:
  **{best_e_learn:.0f}**).
  This can happen with too few relaxations or a learning rate that's too
  high. Try adjusting the parameters.
  """)
        st.markdown("""
  Note: conversely, learning can succeed when the learning rate is low.
  In the limit of low learning rates — such that the system can visit a
  sufficient sample of attractors between weight updates — Hebbian
  updates accumulate reliable recurring sub-patterns rather than
  overfitting transient fluctuations.
  """)
    else:
        st.markdown(f"""
  **Hebbian learning found the best solution** at **{best_e_learn:.0f}**
  (baseline: {best_e_base:.0f}). In the last {tail} relaxations it
  converged to just ~{unique_learn_tail} distinct energy level(s).
  """)
  
    st.caption(
        "All numbers above are computed directly from this run. "
        "Re-running will generate a new random problem and produce "
        "different numbers."
    )
  
    # ─── E^α_0 plot: true energy over relaxations ────────────
    st.divider()
    st.subheader("True energy over relaxations")
  
    st.markdown(r"""
  This plot shows $E^{\alpha}_0$ — the energy of each relaxation's final
  state measured against the **original** constraint matrix $\alpha$
  (the one generated before any learning):
  
  $$E^{\alpha}_0 = -\sum_{i<j} \alpha_{ij}\, s_i\, s_j$$
  
  This is the quantity we actually want to minimise. During Hebbian learning
  the network's internal weights $W$ drift away from $\alpha$, but we always
  score the result against the **true** problem. A downward trend in the
  learning curve means learning is genuinely improving the solutions, not
  just optimising the modified weights.
  """)

    # ─── Scatter: energy over time ───────────────────────────
    # Scale marker size and transparency so the plot stays readable
    # even with tens of thousands of relaxations.  At 300 relaxations
    # the defaults are s=12 / alpha=0.5; at 50,000 they shrink to
    # s≈2 / alpha≈0.08 so individual dots remain distinguishable.
    _dot_size = max(2, min(14, 4200 / num_relaxations))
    _dot_alpha = max(0.25, min(0.7, 300 / num_relaxations))

    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 4))
    xs = np.arange(1, num_relaxations + 1)
    ax_scatter.scatter(xs, energies_base, s=_dot_size, alpha=_dot_alpha,
                       color='#D95F02', label='No learning', zorder=2,
                       rasterized=num_relaxations > 2000)
    ax_scatter.scatter(xs, energies_learn, s=_dot_size, alpha=_dot_alpha,
                       color='#1B7837', label='With learning', zorder=3,
                       rasterized=num_relaxations > 2000)
    ax_scatter.set_xlabel('Relaxation number')
    ax_scatter.set_ylabel(r'$E^{\alpha}_0$  (true energy)')
    _tau_mult = r.get('tau_multiplier', st.session_state.get('tau_multiplier', 10))
    _delta_val = r.get('delta', st.session_state.get('delta', 0.00025))
    _same_tag = "  |  same problem" if r.get('same_problem', False) else ""
    if r['n_modules'] == 0:
        _subtitle = (f"N={N}  |  unstructured  |  "
                     f"\u03c4={_tau_mult}\u00d7N={_tau_mult * N:,}  |  "
                     f"\u03b4={_delta_val}  |  "
                     f"relaxations={num_relaxations}{_same_tag}")
    else:
        _subtitle = (f"N={N} ({r['n_modules']}\u00d7{r['module_size']})  |  "
                     f"\u03c4={_tau_mult}\u00d7N={_tau_mult * N:,}  |  "
                     f"\u03b4={_delta_val}  |  "
                     f"relaxations={num_relaxations}{_same_tag}")
    ax_scatter.set_title('Attractor energy over the course of learning\n'
                         + _subtitle, fontsize=10)
    ax_scatter.legend(fontsize=8, loc='upper right')
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    fig_scatter.tight_layout()
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

    _tau_m = st.session_state.get('tau_multiplier', 10)
    _tau = _tau_m * N

    # ── Dynamic convergence analysis ─────────────────────────
    _el = np.array(energies_learn)
    _eb = np.array(energies_base)
    _base_spread = float(np.std(_eb))
    _base_mean_e = float(np.mean(_eb))

    # Find where learning "converges": rolling std over a window drops
    # below 10% of the baseline spread.  Scan from the start.
    _win = max(10, num_relaxations // 15)
    _conv_relax = None  # relaxation index where convergence detected
    _threshold = _base_spread * 0.10
    for _ci in range(0, len(_el) - _win + 1):
        if np.std(_el[_ci : _ci + _win]) < _threshold:
            _conv_relax = _ci
            break

    _conv_fraction = _conv_relax / num_relaxations if _conv_relax is not None else None
    _final_learn_mean = float(np.mean(_el[-_win:]))
    _final_learn_std = float(np.std(_el[-_win:]))
    _improvement_pct = (best_e_base - best_e_learn) / abs(best_e_base) * 100 if best_e_base != 0 else 0
    _mean_improvement_pct = (_base_mean_e - float(np.mean(_el))) / abs(_base_mean_e) * 100 if _base_mean_e != 0 else 0

    # Baseline trend (should be flat; flag if not)
    _base_first_half = float(np.mean(_eb[:num_relaxations // 2]))
    _base_second_half = float(np.mean(_eb[num_relaxations // 2:]))
    _base_drift = abs(_base_second_half - _base_first_half)

    # Build the descriptive text
    _desc_parts = []

    _desc_parts.append(
        f"Each dot is one relaxation's final energy scored against the "
        f"original problem. There are **{num_relaxations}** dots per series "
        f"— one for each relaxation. "
        f"Two improvement metrics are reported: **best-of-N** compares "
        f"the single lowest dot from each series; **mean** compares "
        f"the average across all dots. When blue dots look consistently "
        f"lower but best-of-N is small, it means baseline got lucky on "
        f"one relaxation — the mean improvement captures learning's "
        f"real advantage in *reliability*."
    )

    # Orange dots analysis
    if _base_drift < _base_spread * 0.1:
        _desc_parts.append(
            f"The **orange dots** (baseline) are scattered in a band around "
            f"**{_base_mean_e:.0f}** (std ≈ {_base_spread:.0f}) with no trend "
            f"— as expected for fixed weights."
        )
    else:
        _desc_parts.append(
            f"The **orange dots** (baseline) are centred around "
            f"**{_base_mean_e:.0f}** (std ≈ {_base_spread:.0f}). "
            f"There is some drift between halves ({_base_first_half:.0f} → "
            f"{_base_second_half:.0f}), likely from randomness in this "
            f"particular problem instance."
        )

    # Blue dots analysis — purely descriptive
    if _conv_relax is not None:
        _conv_label = "early" if _conv_fraction < 0.33 else ("mid-run" if _conv_fraction < 0.66 else "late")
        _conv_text = (
            f"The **blue dots** (with learning) converged **{_conv_label}** — "
            f"by around relaxation {_conv_relax} (roughly "
            f"{_conv_fraction * 100:.0f}% of the way through), the "
            f"energies collapsed to a tight band near "
            f"**{_final_learn_mean:.0f}** (std ≈ {_final_learn_std:.1f}). "
            f"Before convergence, each relaxation was landing in a "
            f"different local minimum; after it, the Hebbian weight "
            f"changes had reshaped the landscape enough that almost every "
            f"relaxation fell into the same basin. "
        )
        if _conv_fraction < 0.33:
            _conv_text += (
                f"Early convergence means learning reshaped the landscape "
                f"quickly — most of the remaining relaxations were spent "
                f"revisiting the same attractor rather than exploring "
                f"further. Note that convergence timing varies "
                f"substantially between runs even on the same problem, "
                f"because it depends on which random trajectories the "
                f"network happened to take. "
            )
        elif _conv_fraction >= 0.66:
            _conv_text += (
                f"Late convergence means the network spent most of its "
                f"relaxation budget still exploring — many different "
                f"attractors were sampled before the landscape settled. "
                f"Note that convergence timing varies substantially "
                f"between runs even on the same problem, because it "
                f"depends on which random trajectories the network "
                f"happened to take. "
            )
        _conv_text += (
            f"The best-of-N improvement was **{_improvement_pct:+.1f}%**; "
            f"the mean improvement was **{_mean_improvement_pct:+.1f}%**."
        )
        _desc_parts.append(_conv_text)
    else:
        _desc_parts.append(
            f"The **blue dots** (with learning) did not clearly converge "
            f"to a single energy level within {num_relaxations} "
            f"relaxations. The final {_win} relaxations had mean energy "
            f"**{_final_learn_mean:.0f}** (std ≈ {_final_learn_std:.1f}), "
            f"compared to baseline std ≈ {_base_spread:.0f}. "
            f"The best-of-N improvement was **{_improvement_pct:+.1f}%**; "
            f"the mean improvement was **{_mean_improvement_pct:+.1f}%**. "
            f"Learning may need more relaxations, or a slower learning "
            f"rate, to fully reshape the landscape."
        )

    _desc_parts.append(
        f"Watson et al.'s central insight is that learning works by "
        f"extracting **regularities across many mediocre attempts**. "
        f"The convergence point and the size of the improvement vary "
        f"between random problem instances."
    )

    if r.get('same_problem', False):
        _desc_parts.append(
            f"This run used the **same constraint matrix** as the "
            f"previous experiment. The **Re-run same problem** button "
            f"isolates the stochasticity of the *search process* from "
            f"the stochasticity of the *problem*: the constraint matrix "
            f"is fixed, but each relaxation still starts from a fresh "
            f"random state with a random update order, so results will "
            f"vary between re-runs. Any difference in convergence speed "
            f"or improvement % is therefore due to the random "
            f"trajectories and the current settings (τ, δ, relaxations), "
            f"not a different problem. Try varying the learning rate (δ) "
            f"or number of relaxations and re-running to see how they "
            f"affect performance on this specific problem."
        )
    else:
        _desc_parts.append(
            f"Each run generates a "
            f"**different random constraint matrix**, so comparing "
            f"convergence speed against improvement % across runs is "
            f"apples-to-oranges: the improvement depends not just on how "
            f"well learning did, but on how hard the particular problem "
            f"was for baseline. An 'easy' random problem can produce fast "
            f"convergence *and* a large improvement; a 'hard' one can "
            f"produce slow convergence *and* a small improvement. To "
            f"fairly test whether more exploration helps, use the "
            f"**Re-run same problem** button and vary the learning rate "
            f"(δ) — or use the **multi-trial analysis**, which averages "
            f"across many random instances to control for this variation."
        )

    st.markdown("\n\n".join(_desc_parts))
  
    # ─── Watson Fig. 1 — attractor energy distributions ───────
    st.pyplot(_watson_fig1_real(energies_base, energies_learn))
  
    st.markdown(r"""
  **Reading the figure:**
  
  - **Each dot** is a single relaxation's final attractor energy
    $E^{\alpha}_0$. Every time the network is released from a random
    starting state and allowed to settle, it lands in some attractor —
    the dot shows how good that attractor was, scored against the
    **original** problem $\alpha$.
  
  - **The red curve** beside each strip is a kernel-density estimate (KDE)
    — a smoothed histogram. Where the curve bulges out, many relaxations
    landed at similar energies; where it is thin, outcomes were rare.
  
  - **Panel (a)** shows the baseline: no learning, original connections
    only. The spread of dots tells you how much variety there is among
    the attractors the network finds at random.
  
  - **Panel (b)** shows the first third of relaxations *with* Hebbian
    learning active. The weight matrix $W$ is beginning to drift away
    from $\alpha$, merging some basins — the spread should start to
    narrow.
  
  - **Panel (c)** shows the last third. By now the learned weights have
    simplified the landscape substantially. If learning is working, the
    dots cluster at lower (more negative) energies and the distribution
    is tighter — the network reliably finds better solutions to the
    original problem.
  
  More negative = better (lower energy = more constraints satisfied).
  """)
  

# ─── MULTI-TRIAL ────────────────────────────────────────────
def _run_multi_trial(run):
  if not run:
    return

  n_modules = st.session_state.n_modules
  module_size = st.session_state.module_size
  intra_strength = st.session_state.intra_strength
  inter_strength = st.session_state.inter_strength
  positive_bias = st.session_state.positive_bias
  num_relaxations = st.session_state.num_relaxations
  delta = st.session_state.delta
  tau_multiplier = st.session_state.tau_multiplier
  num_trials = st.session_state.num_trials

  if n_modules == 0:
      N = module_size
  else:
      N = n_modules * module_size

  TAU = tau_multiplier * N
  DELTA_PER_UPDATE = delta / TAU

  master_rng = np.random.default_rng()
  trial_seeds = master_rng.integers(0, 2**63, size=num_trials)

  best_base_list = []
  best_learn_list = []
  improvements = []
  mean_improvements = []
  learning_won_count = 0

  progress = st.progress(0, text="Running multi-trial analysis...")
  for i, seed in enumerate(trial_seeds):
      progress.progress((i + 1) / num_trials,
                         text=f"Trial {i + 1} / {num_trials}")

      # Each trial gets a new random problem instance
      problem_rng = np.random.default_rng(seed)
      if n_modules == 0:
          uniform_strength = (intra_strength + inter_strength) / 2
          alpha = generate_modular_problem(
              1, N, uniform_strength, uniform_strength,
              positive_bias, problem_rng,
          )
      else:
          alpha = generate_modular_problem(
              n_modules, module_size, intra_strength,
              inter_strength, positive_bias, problem_rng,
          )

      # Watson: "The same 300 random initial conditions and the same
      # random order of state updates are used for the learning and
      # non-learning runs of each trial."
      run_seed = problem_rng.integers(0, 2**63)
      rng_base = np.random.default_rng(run_seed)
      rng_learn = np.random.default_rng(run_seed)

      _e_base, best_e_base, _s_base, _ = run_baseline(
          alpha, num_relaxations, rng_base, tau=TAU,
      )
      _e_learn, best_e_learn, _s_learn, _ = run_with_learning(
          alpha, num_relaxations, DELTA_PER_UPDATE, rng_learn, tau=TAU,
      )

      best_base_list.append(best_e_base)
      best_learn_list.append(best_e_learn)
      imp = ((best_e_base - best_e_learn) / abs(best_e_base) * 100
             if best_e_base != 0 else 0.0)
      improvements.append(imp)

      _mb = float(np.mean(_e_base))
      _ml = float(np.mean(_e_learn))
      mean_imp = ((_mb - _ml) / abs(_mb) * 100) if _mb != 0 else 0.0
      mean_improvements.append(mean_imp)

      if best_e_learn < best_e_base:
          learning_won_count += 1

  progress.empty()

  st.session_state._multi_run = st.session_state.get('_multi_run', 0) + 1
  st.session_state._multi_just_ran = True

  st.session_state.multi_trial_results = {
      'num_trials': num_trials,
      'n_modules': n_modules,
      'module_size': module_size,
      'N': N,
      'num_relaxations': num_relaxations,
      'best_base': np.array(best_base_list),
      'best_learn': np.array(best_learn_list),
      'improvements': np.array(improvements),
      'mean_improvements': np.array(mean_improvements),
      'learning_won_count': learning_won_count,
  }


def _render_multi_trial():
  if 'multi_trial_results' not in st.session_state:
    return

  _mt_id = st.session_state.get('_multi_run', 0)
  with st.expander(f"Multi-trial analysis results (run {_mt_id})", expanded=True):
    mt = st.session_state.multi_trial_results
    num_trials = mt['num_trials']
    best_base = mt['best_base']
    best_learn = mt['best_learn']
    improvements = mt['improvements']
    mean_improvements = mt.get('mean_improvements', improvements)
    won = mt['learning_won_count']
  
    tied = int(np.sum(best_learn == best_base))
    lost = num_trials - won - tied
  
    st.markdown(
        f"**{num_trials}** trials, each on a different random problem "
        f"instance ({mt['N']} switches"
        + (f", {mt['n_modules']} modules of {mt['module_size']}"
           if mt['n_modules'] > 0 else ", unstructured")
        + f", {mt['num_relaxations']} relaxations per run). "
        f"Within each trial the baseline and learning runs used the "
        f"**same** random initial conditions and update order."
    )
  
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Learning won", f"{won} / {num_trials}",
                   delta=f"{won / num_trials * 100:.0f}%")
    with c2:
        st.metric("Tied", f"{tied}")
    with c3:
        st.metric("Learning lost", f"{lost}")
    with c4:
        st.metric("Mean best-of-N impr.",
                   f"{np.mean(improvements):+.1f}%")
    with c5:
        st.metric("Mean mean impr.",
                   f"{np.mean(mean_improvements):+.1f}%")
  
    # ── Histogram of improvements ──
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, data, label in zip(
        axes,
        [improvements, mean_improvements],
        ['Best-of-N improvement (%)', 'Mean improvement (%)'],
    ):
        ax.hist(data, bins='auto', color='steelblue', edgecolor='white',
                alpha=0.85)
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.axvline(np.mean(data), color='crimson', linewidth=1.2,
                   label=f"mean = {np.mean(data):+.1f}%")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of trials")
        ax.legend(fontsize=8)
    axes[0].set_title("Best-of-N improvement across trials")
    axes[1].set_title("Mean improvement across trials")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
  
    st.markdown(
        "Each bar counts how many trials fell in that improvement range. "
        "Positive values (right of the dashed line) mean learning found a "
        "lower-energy solution than the baseline; negative means it did "
        "worse. The red line marks the mean. **Best-of-N** compares the "
        "single best relaxation from each run; **Mean** compares the "
        "average across all relaxations."
    )
  
    # ── Summary statistics table ──
    import pandas as pd
    summary = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Std dev', 'Min', 'Max'],
        'Baseline best energy': [
            f"{np.mean(best_base):.1f}",
            f"{np.median(best_base):.1f}",
            f"{np.std(best_base):.1f}",
            f"{np.min(best_base):.1f}",
            f"{np.max(best_base):.1f}",
        ],
        'Learning best energy': [
            f"{np.mean(best_learn):.1f}",
            f"{np.median(best_learn):.1f}",
            f"{np.std(best_learn):.1f}",
            f"{np.min(best_learn):.1f}",
            f"{np.max(best_learn):.1f}",
        ],
        'Improvement % (best-of-N)': [
            f"{np.mean(improvements):+.1f}",
            f"{np.median(improvements):+.1f}",
            f"{np.std(improvements):.1f}",
            f"{np.min(improvements):+.1f}",
            f"{np.max(improvements):+.1f}",
        ],
        'Improvement % (mean)': [
            f"{np.mean(mean_improvements):+.1f}",
            f"{np.median(mean_improvements):+.1f}",
            f"{np.std(mean_improvements):.1f}",
            f"{np.min(mean_improvements):+.1f}",
            f"{np.max(mean_improvements):+.1f}",
        ],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)
  
    st.caption(
        "Energy values are the standard Hopfield energy "
        "E\u1d45\u2080 = \u2212\u00bd \u2211 \u03b1\u1d62\u2c7c s\u1d62s\u2c7c. "
        "More negative = more constraints satisfied = better. "
        "The absolute magnitude depends on problem size and connection "
        "strengths, so compare across columns (baseline vs learning) "
        "rather than across rows. The Improvement % column normalises "
        "the comparison: positive means learning found a lower-energy "
        "(better) solution."
    )
  
  
# Detect pending computation from a PREVIOUS rerun (pass 2).
# This runs BEFORE rendering so the spinner/progress show on the
# (already clean) page.
_execute = st.session_state.pop('_execute_run', None)
if _execute == 'single':
    _run_experiment(True)
    st.rerun()
elif _execute == 'same':
    _run_experiment(True, same_problem=True)
    st.rerun()
elif _execute == 'multi':
    _run_multi_trial(True)
    st.rerun()

_render_results()
_render_multi_trial()
_render_info_sections()

# If EITHER button was pressed THIS rerun, the on_click already cleared
# old results.  The page above rendered clean (no results).  Now we set
# a flag and call st.rerun() — Streamlit sends the current (clean) page
# to the browser BEFORE starting the next run, which will detect the
# flag and do the actual computation.
if _run_btn:
    st.session_state['_execute_run'] = 'single'
    st.rerun()
elif _rerun_btn:
    st.session_state['_execute_run'] = 'same'
    st.rerun()
elif _multi_btn:
    st.session_state['_execute_run'] = 'multi'
    st.rerun()