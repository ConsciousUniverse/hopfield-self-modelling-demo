# Self-Modelling: How a Network Learns to Optimise

An interactive Streamlit demo of the mechanism described in:

> **Watson, R. A., Buckley, C. L., & Mills, R. (2011).**
> *Optimisation in Self-modelling Complex Adaptive Systems.*
> Complexity, 16(5), 17–26.
> **[doi:10.1002/cplx.20346](https://doi.org/10.1002/cplx.20346)**

The paper shows that a Hopfield network with concurrent Hebbian learning can solve modular constraint problems far better than random search — not by recalling past solutions, but by *generalising* from many mediocre experiences into something superior.

This demo lets you run the experiment yourself, explore the parameters, and see the results in real time.

---

## How it works

We have **150 binary switches** (configurable), each ON (+1) or OFF (−1). Every pair of switches is joined by a **connection** — a positive connection means "prefer to be the same," a negative one means "prefer to be different." The size of the number says how strong the preference is.

### What is the network trying to do?

No one tells the network the answer. There are thousands of connections that contradict each other — no arrangement can satisfy them all simultaneously. The network's job is to find the arrangement that **satisfies as many preferences as possible, weighted by their strengths**. This is measured by the **energy**: lower energy = more preferences satisfied.

This is *not* pattern recall. It is a combinatorial optimisation puzzle over 2^N possible arrangements.

### Modular structure

The switches are arranged in **groups** (default: 30 groups of 5). Connections within a group are **strong** (magnitude 1.0); connections between groups are **weak** (magnitude 0.01). This creates a nearly-decomposable modular structure — exactly the kind of problem where Hebbian learning shines.

### The three phases

1. **Without learning (baseline)** — The network tries many random starting arrangements. Each "relaxation" picks one switch at a time, updates it based on all its neighbours, and repeats for τ = 10N steps. Energy is measured at the end, then the arrangement is thrown away and we start fresh. The connections never change. Because the same fixed connections keep producing similar outcomes, certain patterns recur across relaxations — these regularities are what Phase 2 exploits.

2. **Flat Hebbian learning** — The same process, but now at every single state update, every connection in the whole network gets a tiny nudge: pairs of switches that are currently the same get a slightly stronger "agree" connection; pairs that differ get a slightly stronger "disagree" connection. The switch arrangement is still thrown away between relaxations, but the adjusted connection strengths are kept. Over hundreds of relaxations, the nudges accumulate fastest for recurring patterns, gradually reshaping the energy landscape until the network reliably finds solutions **better than any it has seen before**.

To see what happens without modular structure, set the number of modules to **0**. This creates an unstructured problem with uniform connection strengths — every pair of switches is connected equally strongly. Because there are no recurring sub-patterns for learning to latch onto, the improvement is typically much smaller. This demonstrates that **modular structure is what makes Hebbian learning effective**.

### The plot: true energy E<sup>α</sup><sub>0</sub>

The key output is a plot of E<sup>α</sup><sub>0</sub> over relaxations — the energy of each final state measured against the **original** constraint matrix α (before any learning modified the weights). This is the quantity we actually want to minimise (Watson Eq. 4). A downward trend in the learning curve means learning is genuinely improving solutions to the original problem, not just optimising modified weights.

---

## Run locally

```bash
pip install streamlit numpy pandas pillow
streamlit run hebbian_demo_4.py
```

Or with Pipenv from the parent directory:

```bash
pipenv run streamlit run streamlit/hebbian_demo_4.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, point it at your repo, set the main file to `hebbian_demo_4.py`.
4. Deploy.

The `requirements.txt` in this folder lists the dependencies Streamlit Cloud needs.

## Files

| File | Purpose |
|------|---------|
| `hebbian_demo_4.py` | Streamlit UI — parameters, visualisation, descriptions |
| `hopfield_algorithm.py` | Pure algorithm module — relaxation, Hebbian learning, energy calculation |
| `requirements.txt` | pip dependencies for Streamlit Cloud deployment |
