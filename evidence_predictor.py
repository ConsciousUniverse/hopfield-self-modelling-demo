import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hopfield_algorithm import run_baseline, run_with_learning

# --- Page Configuration ---
st.set_page_config(page_title="Modular Hebbian Predictor", layout="wide")

# --- Logic: Matrix Construction ---
def create_modular_evidence_matrix(evidence_data, hypothesis_impacts, intra_val, inter_val):
    """
    Uses the Watson, Buckley & Mills principles of modularity:
    - Intra-strength: Applied between evidence in the SAME module.
    - Inter-strength: Applied between evidence in DIFFERENT modules.
    """
    num_nodes = len(evidence_data) + 1
    W = np.zeros((num_nodes, num_nodes))
    
    # 1. Hypothesis (Node 0) to Evidence (Nodes 1..N)
    for i, impact in enumerate(hypothesis_impacts):
        W[0, i+1] = impact
        W[i+1, 0] = impact
        
    # 2. Evidence-to-Evidence (Inter-Clue Connections)
    for i in range(len(evidence_data)):
        for j in range(i + 1, len(evidence_data)):
            # Get modules safely to avoid KeyError from old session data
            mod_i = evidence_data[i].get('module', 'default')
            mod_j = evidence_data[j].get('module', 'default')
            
            same_module = (mod_i == mod_j)
            strength = intra_val if same_module else inter_val
            
            # Coordination: Do these two pieces of evidence agree?
            # If both are supportive (+1/+1) or both contradictory (-1/-1), 
            # they should have a positive bond to form a consistent 'schema'.
            direction = 1.0 if np.sign(hypothesis_impacts[i]) == np.sign(hypothesis_impacts[j]) else -1.0
            
            W[i+1, j+1] = direction * strength
            W[j+1, i+1] = direction * strength
                
    return W

# --- UI: Header ---
st.title("🧠 Modular Hebbian Evidence Predictor")
st.markdown("""
Based on **Watson, Buckley & Mills (2011)**. This system treats prediction as a 
search for a **Global Minimum** in an energy landscape. By grouping evidence 
into modules, we allow the system to generalize and find the most consistent narrative.
""")

# --- Sidebar: Input & Controls ---
with st.sidebar:
    st.header("1. Global Constraints")
    intra = st.slider("Intra-Module Strength", 0.1, 5.0, 1.0, 
                      help="Bond strength between evidence in the same group.")
    inter = st.slider("Inter-Module Strength", 0.0, 1.0, 0.01, 
                      help="Bond strength between different groups.")
    
    st.divider()
    
    st.header("2. Hypothesis")
    h_name = st.text_input("Predicting Event:", "Event X")
    
    st.divider()
    
    st.header("3. Add Evidence")
    if 'evidence' not in st.session_state or not isinstance(st.session_state.evidence, list):
        st.session_state.evidence = []

    with st.form("evidence_form", clear_on_submit=True):
        e_name = st.text_input("Evidence Name")
        e_mod = st.text_input("Module (e.g., 'Forensics')")
        e_type = st.selectbox("Type", ["Supportive", "Contradictory"])
        e_weight = st.slider("Weight/Certainty", 1, 10, 5)
        
        if st.form_submit_button("Add Evidence"):
            if e_name and e_mod:
                st.session_state.evidence.append({
                    "name": e_name, 
                    "module": e_mod, 
                    "type": e_type, 
                    "strength": e_weight
                })
            else:
                st.error("Please provide a name and module.")

    st.divider()
    
    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        st.session_state.evidence = []
        st.rerun()

# --- Main Interface ---
if not st.session_state.evidence:
    st.info("👈 Add pieces of evidence in the sidebar to begin the experiment.")
else:
    col_table, col_viz = st.columns([1, 1])
    
    with col_table:
        st.subheader("Current Evidence Stack")
        df = pd.DataFrame(st.session_state.evidence)
        st.table(df)

    # Prepare Weights
    impacts = [e['strength'] if e['type'] == "Supportive" else -e['strength'] 
               for e in st.session_state.evidence]
    
    W = create_modular_evidence_matrix(st.session_state.evidence, impacts, intra, inter)
    
    st.divider()
    
    st.subheader("4. Run Prediction")
    n_relax = st.number_input("Simulation Iterations", 100, 5000, 1000)
    
    if st.button("Execute 'Self-Modelling' Simulation", type="primary"):
        rng = np.random.default_rng()
        
        # 1. Baseline (Raw Weights)
        _, _, _, states_base = run_baseline(W, n_relax, rng)
        
        # 2. Hebbian Learning (Watson et al. Method)
        # delta=0.005 allows the basins to reshape without collapsing immediately
        _, _, _, states_learn = run_with_learning(W, n_relax, delta=0.005, rng=rng)
        
        # Calculate Hypothesis (Node 0) outcome frequency
        h_base = [s[0] for s in states_base].count(1)
        h_learn = [s[0] for s in states_learn].count(1)
        
        prob_base = (h_base / n_relax) * 100
        prob_learn = (h_learn / n_relax) * 100
        
        # Results Display
        res_left, res_right = st.columns(2)
        
        res_left.metric("Baseline Probability", f"{prob_base:.1f}%")
        res_right.metric("Self-Modelled Probability", f"{prob_learn:.1f}%", 
                         delta=f"{prob_learn - prob_base:.1f}% confidence")
        
        # Final Verdict
        st.divider()
        if prob_learn > 50:
            st.success(f"**PREDICTION:** It is highly likely that **{h_name}** occurred.")
        else:
            st.error(f"**PREDICTION:** It is likely that **{h_name}** did NOT occur.")

        # Distribution Chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist([s[0] for s in states_learn], bins=[-1.5, -0.5, 0.5, 1.5], 
                rwidth=0.4, color='#3372B0', alpha=0.8)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(['False', 'True'])
        ax.set_title("Distribution of Hypothesis Attractors")
        st.pyplot(fig)

        st.info("""
        **How to read this:** If the probability shifted significantly after learning, 
        the network successfully 'generalised' the evidence structure, finding a 
        globally consistent conclusion despite local contradictions.
        """)