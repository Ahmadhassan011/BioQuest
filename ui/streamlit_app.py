"""
Streamlit UI for BioQuest.

Interactive web interface for molecule optimization.
"""

import json

import streamlit as st

from src.inference import MoleculePredictor, ModelNotLoadedError

st.set_page_config(page_title="BioQuest", page_icon="M")

st.title("BioQuest")
st.markdown("AI-Driven Drug Discovery Platform")

if "results" not in st.session_state:
    st.session_state.results = None

if "predictor" not in st.session_state:
    st.session_state.predictor = None

st.sidebar.title("Configuration")

protein_seq = st.sidebar.text_input(
    "Protein Sequence",
    value="MKFLILLFSLLGICLPAVGGKKLAT",
    help="Target protein sequence for drug binding",
)

seeds_input = st.sidebar.text_area(
    "Seed Molecules (SMILES)",
    value="CCO\nc1ccccc1\nCC(=O)O",
    help="Enter SMILES strings, one per line",
)

max_iterations = st.sidebar.slider("Max Iterations", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 5, 100, 20)

st.sidebar.title("Objective Weights")
w_affinity = st.sidebar.slider("Binding Affinity", 0.0, 1.0, 0.4)
w_toxicity = st.sidebar.slider("Low Toxicity", 0.0, 1.0, 0.3)
w_qed = st.sidebar.slider("Drug-likeness (QED)", 0.0, 1.0, 0.2)
w_sa = st.sidebar.slider("Synthetic Accessibility", 0.0, 1.0, 0.1)

objectives = {
    "affinity": w_affinity,
    "toxicity": w_toxicity,
    "qed": w_qed,
    "sa": w_sa,
}

if st.button("Run Optimization"):
    if not protein_seq or len(protein_seq) < 5:
        st.error("Please enter a valid protein sequence (min 5 chars)")
    else:
        seeds = [s.strip() for s in seeds_input.split("\n") if s.strip()]
        if not seeds:
            st.error("Please enter at least one valid SMILES")
        else:
            with st.spinner("Initializing predictor..."):
                if st.session_state.predictor is None:
                    try:
                        st.session_state.predictor = MoleculePredictor(
                            protein_sequence=protein_seq,
                            use_gpu=False,
                            models_dir="artifacts/models",
                        )
                    except ModelNotLoadedError as e:
                        st.error(f"Failed to load models: {e}. Please ensure trained models exist in artifacts/models/")
                        st.stop()

            predictor = st.session_state.predictor
            st.info(f"Evaluating {len(seeds)} molecules...")

            all_results = []
            for smiles in seeds:
                try:
                    props = predictor.predict_all_properties(smiles)
                    score = predictor.score_molecule(smiles, objectives)
                    all_results.append({
                        "smiles": smiles,
                        **props,
                        "composite_score": score,
                    })
                except Exception as e:
                    st.warning(f"Failed to process {smiles}: {e}")

            if all_results:
                all_results.sort(key=lambda x: x["composite_score"], reverse=True)
                st.session_state.results = all_results

                st.success(f"Evaluated {len(all_results)} molecules")

if st.session_state.results:
    st.subheader("Results")

    top_results = st.session_state.results[:10]
    for i, mol in enumerate(top_results):
        with st.expander(f"#{i+1}: Score={mol['composite_score']:.3f}"):
            st.markdown(f"**SMILES:** `{mol['smiles']}`")
            cols = st.columns(3)
            cols[0].metric("Affinity", f"{mol['affinity']:.4f}")
            cols[1].metric("Toxicity", f"{mol['toxicity']:.4f}")
            cols[2].metric("QED", f"{mol['qed']:.3f}")

            cols = st.columns(3)
            cols[0].metric("SA", f"{mol['sa']:.3f}")
            cols[1].metric("LogP", f"{mol['logp']:.2f}")
            cols[2].metric("MW", f"{mol['mw']:.1f}")

    st.download_button(
        "Download Results (JSON)",
        data=json.dumps(st.session_state.results, indent=2),
        file_name="bioquest_results.json",
        mime="application/json",
    )