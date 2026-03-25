"""
UI Module: Streamlit-based interactive interface for BioQuest.

Features:
- Input widgets for protein sequence, seed molecules, and objectives
- Real-time optimization progress visualization
- Interactive parameter tuning
- Convergence and Pareto front charts
- Results export and analysis
- Direct backend integration without subprocess calls
"""

import logging
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List
import json
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class BioQuestStreamlitUI:
    """Streamlit UI for BioQuest."""

    def __init__(self):
        """Initialize Streamlit UI."""
        self.config_streamlit()

    def config_streamlit(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="BioQuest - AI Drug Discovery",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS
        st.markdown(
            """
        <style>
        .stButton>button {
            width: 100%;
        }
        .stSpinner {
            text-align: center;
        }
        .metric-card {
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def validate_protein_sequence(self, sequence: str) -> tuple[bool, str]:
        """
        Validate protein sequence.

        Args:
            sequence: Protein sequence to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        sequence = sequence.upper().strip()

        if not sequence:
            return False, "Protein sequence cannot be empty"

        if len(sequence) < 5:
            return False, "Protein sequence must be at least 5 amino acids"

        invalid_aa = [aa for aa in sequence if aa not in valid_aa]
        if invalid_aa:
            return False, f"Invalid amino acids: {set(invalid_aa)}"

        return True, ""

    def validate_smiles(self, smiles: str) -> tuple[bool, str]:
        """
        Validate SMILES string.

        Args:
            smiles: SMILES string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        from rdkit import Chem

        smiles = smiles.strip()
        if not smiles:
            return False, "SMILES cannot be empty"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES: {smiles}"

        return True, ""

    def render_sidebar(self) -> str:
        """Render the sidebar navigation."""
        st.sidebar.title("BioQuest Navigation")
        st.sidebar.markdown("---")

        page = st.sidebar.radio(
            "Go to",
            ["Configuration", "Run Optimization", "Results"],
            key="page_navigation",
        )

        st.sidebar.markdown("---")
        st.sidebar.info(
            "**BioQuest** is an AI-driven platform for autonomous drug discovery, "
            "combining multi-agent orchestration, hybrid molecule generation, "
            "and multi-objective optimization."
        )

        st.sidebar.markdown("---")
        if "results" in st.session_state and st.session_state.results:
            st.sidebar.success("Results available!")
            if st.sidebar.button("Download Results"):
                results_json = json.dumps(st.session_state.results, indent=2)
                st.sidebar.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"bioquest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

        return page

    def render_header(self) -> None:
        """Render the main application header."""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("BioQuest")
            st.markdown("An AI-powered platform for accelerated *de novo* drug design.")
        with col2:
            st.metric(
                "Status",
                "Ready"
                if "config" not in st.session_state or st.session_state.get("config")
                else "Configured",
            )

    def render_config_page(self) -> None:
        """Render the configuration page."""
        st.warning(
            "This AI system is intended for **educational and research purposes only**. The generated molecules are predictions and have **not been experimentally validated**. Use caution and consult qualified professionals before considering any practical applications. The developers **do not accept any responsibility** for the use or misuse of this software."
        )
        st.header("Step 1: Configure Your Experiment")

        with st.form(key="config_form", clear_on_submit=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Target Protein")
                protein_seq = st.text_area(
                    "Enter protein sequence (amino acids):",
                    value=st.session_state.get("protein_seq", "MKFLVLLACFAATVAGA"),
                    height=100,
                    help="Provide the single-letter amino acid sequence of the target protein. Valid: ACDEFGHIKLMNPQRSTVWY",
                )

            with col2:
                st.subheader("Seed Molecules")
                seed_input = st.text_area(
                    "Enter seed molecule SMILES (one per line):",
                    value=st.session_state.get(
                        "seed_input",
                        "CC(C)Cc1ccc(cc1)C(C)C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                    ),
                    height=100,
                    help="Provide initial molecules to guide the generation process.",
                )

            st.subheader("Optimization Objectives")
            col1, col2 = st.columns(2)

            with col1:
                affinity_weight = st.slider(
                    "Affinity (Binding Strength)",
                    0.0,
                    1.0,
                    value=st.session_state.get("affinity_weight", 0.4),
                    help="Higher values prioritize stronger binding to the target.",
                )
                toxicity_weight = st.slider(
                    "Toxicity (Lower is Better)",
                    0.0,
                    1.0,
                    value=st.session_state.get("toxicity_weight", 0.2),
                    help="Higher values prioritize minimizing predicted toxicity.",
                )

            with col2:
                qed_weight = st.slider(
                    "QED (Drug-Likeness)",
                    0.0,
                    1.0,
                    value=st.session_state.get("qed_weight", 0.2),
                    help="Higher values prioritize molecules with drug-like properties.",
                )
                sa_weight = st.slider(
                    "SA (Synthesizability)",
                    0.0,
                    1.0,
                    value=st.session_state.get("sa_weight", 0.2),
                    help="Higher values prioritize molecules that are easier to synthesize.",
                )

            # Show total weight
            total_weight = affinity_weight + toxicity_weight + qed_weight + sa_weight
            col1, col2 = st.columns(2)
            with col1:
                if total_weight == 0:
                    st.error("Total weight must be > 0")
                else:
                    pass
                    # st.info(f"Total weight: {total_weight:.2f}")

            with st.expander("Advanced Parameters"):
                max_iterations = st.number_input(
                    "Maximum Iterations",
                    1,
                    500,
                    value=st.session_state.get("max_iterations", 50),
                    help="The maximum number of optimization cycles to run.",
                )
                batch_size = st.number_input(
                    "Molecules per Iteration",
                    10,
                    500,
                    value=st.session_state.get("batch_size", 50),
                    help="The number of new molecules to generate and evaluate in each cycle.",
                )
                use_gpu = st.checkbox(
                    "Use GPU (if available)",
                    value=st.session_state.get("use_gpu", False),
                    help="Accelerate predictions with GPU if available.",
                )

            submitted = st.form_submit_button(
                "Save Configuration", use_container_width=True
            )

            if submitted:
                # Validate inputs
                valid_protein, protein_err = self.validate_protein_sequence(protein_seq)
                if not valid_protein:
                    st.error(f"Invalid protein sequence: {protein_err}")
                    return

                seeds = [s.strip() for s in seed_input.split("\n") if s.strip()]
                if not seeds:
                    st.error("Please provide at least one seed molecule")
                    return

                invalid_seeds = []
                for seed in seeds:
                    valid_smiles, smiles_err = self.validate_smiles(seed)
                    if not valid_smiles:
                        invalid_seeds.append(f"{seed}: {smiles_err}")

                if invalid_seeds:
                    st.error("Invalid SMILES:\n" + "\n".join(invalid_seeds))
                    return

                if total_weight == 0:
                    st.error("Total objective weight must be > 0")
                    return

                # Save to session state for persistence
                st.session_state.protein_seq = protein_seq
                st.session_state.seed_input = seed_input
                st.session_state.affinity_weight = affinity_weight
                st.session_state.toxicity_weight = toxicity_weight
                st.session_state.qed_weight = qed_weight
                st.session_state.sa_weight = sa_weight
                st.session_state.max_iterations = max_iterations
                st.session_state.batch_size = batch_size
                st.session_state.use_gpu = use_gpu

                st.session_state.config = {
                    "protein_sequence": protein_seq,
                    "seeds": seeds,
                    "objectives": {
                        "affinity": affinity_weight / total_weight,
                        "toxicity": toxicity_weight / total_weight,
                        "qed": qed_weight / total_weight,
                        "sa": sa_weight / total_weight,
                    },
                    "max_iterations": max_iterations,
                    "batch_size": batch_size,
                    "use_gpu": use_gpu,
                }

                st.success(
                    "Configuration saved! Click on 'Run Optimization' to proceed."
                )

    def render_run_page(self) -> None:
        """Render the page for running the optimization."""
        st.header("Step 2: Run Optimization")

        if "config" not in st.session_state or not st.session_state.get("config"):
            st.warning(
                "Please configure your experiment on the 'Configuration' page first."
            )
            return

        # Display current configuration
        with st.expander("Current Configuration"):
            config = st.session_state.config
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Protein Length:** {len(config['protein_sequence'])} AA")
                st.write(f"**Seed Molecules:** {len(config['seeds'])}")
            with col2:
                st.write(f"**Max Iterations:** {config['max_iterations']}")
                st.write(f"**Batch Size:** {config['batch_size']}")

            st.write("**Objectives:**")
            for obj, weight in config["objectives"].items():
                st.write(f"  - {obj}: {weight:.2%}")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "Start Optimization", use_container_width=True, type="primary"
            ):
                st.session_state.running = True
                st.session_state.optimization_progress = 0
                st.rerun()

        with col2:
            if st.button("Reset", use_container_width=True):
                st.session_state.running = False
                st.session_state.optimization_progress = 0
                if "results" in st.session_state:
                    del st.session_state.results
                st.rerun()

        with col3:
            if st.button("Modify Config", use_container_width=True):
                st.info(
                    "Use the sidebar to navigate back to Configuration page to modify settings."
                )

        # Run optimization if started
        if st.session_state.get("running"):
            st.info("Optimization is running...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_log = st.empty()

            try:
                # Import components here to avoid circular imports
                import sys

                sys.path.insert(
                    0,
                    "/home/ahmad-hassan/Desktop/ABCX/Semester_05/AI/AI_Drug_Discovery",
                )
                from src.app.main import initialize_components, run_optimization_loop

                status_text.write("Initializing components...")
                components = initialize_components(st.session_state.config)

                status_text.write("Running optimization loop...")
                results = run_optimization_loop(components, st.session_state.config)

                # Save results to session state
                st.session_state.results = results
                st.session_state.running = False

                st.success("Optimization complete! Check the Results page.")

            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                logger.error(f"Optimization error: {traceback.format_exc()}")
                st.session_state.running = False

    def render_convergence_plot(self, metrics: Dict) -> None:
        """Render convergence analysis plot."""
        if not metrics:
            st.warning("No convergence metrics available")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Extract data from nested structure
            history = metrics.get("history", {})
            best_scores = history.get("best_scores", [])
            mean_scores = history.get("mean_scores", [])
            iterations = history.get("iterations", [])
            convergence_info = metrics.get("convergence", {})
            stats = metrics.get("statistics", {})

            # Best score over iterations
            if best_scores and iterations:
                axes[0, 0].plot(
                    iterations, best_scores, marker="o", linewidth=2, color="blue"
                )
                axes[0, 0].set_title("Best Score Over Iterations")
                axes[0, 0].set_xlabel("Iteration")
                axes[0, 0].set_ylabel("Composite Score")
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[0, 0].set_title("Best Score Over Iterations")

            # Average score over iterations
            if mean_scores and iterations:
                axes[0, 1].plot(
                    iterations, mean_scores, marker="s", color="orange", linewidth=2
                )
                axes[0, 1].set_title("Average Score Over Iterations")
                axes[0, 1].set_xlabel("Iteration")
                axes[0, 1].set_ylabel("Composite Score")
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, "No data", ha="center", va="center")
                axes[0, 1].set_title("Average Score Over Iterations")

            # Convergence status
            best_score = stats.get("best_score", 0)
            iterations_since_improvement = convergence_info.get(
                "iterations_since_improvement", 0
            )
            status = f"Best: {best_score:.4f}\nStalled for {iterations_since_improvement} iterations"
            axes[1, 0].text(
                0.5,
                0.5,
                status,
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                family="monospace",
            )
            axes[1, 0].axis("off")
            axes[1, 0].set_title("Convergence Status")

            # Statistics box
            stats_text = (
                f"Total Evaluated: {stats.get('total_evaluated', 0)}\n"
                f"Current Iteration: {stats.get('current_iteration', 0)}\n"
                f"Iterations Since Improvement: {convergence_info.get('iterations_since_improvement', 0)}"
            )
            axes[1, 1].text(
                0.5,
                0.5,
                stats_text,
                ha="center",
                va="center",
                fontsize=11,
                family="monospace",
            )
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Statistics")

            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not render convergence plot: {e}")

    def render_pareto_front(self, pareto_front: List[Dict]) -> None:
        """Render Pareto front visualization."""
        if not pareto_front:
            st.warning("No Pareto front data available")
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            affinity_scores = [m.get("affinity", 0) for m in pareto_front]
            toxicity_scores = [m.get("toxicity", 0) for m in pareto_front]
            composite_scores = [m.get("composite_score", 0) for m in pareto_front]

            scatter = ax.scatter(
                affinity_scores,
                toxicity_scores,
                c=composite_scores,
                cmap="viridis",
                s=100,
                alpha=0.6,
                edgecolors="black",
            )

            ax.set_xlabel("Affinity (Higher is Better)")
            ax.set_ylabel("Toxicity (Lower is Better)")
            ax.set_title("Pareto Front: Trade-offs Between Objectives")
            ax.grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Composite Score")

            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not render Pareto front: {e}")

    def render_results_table(self, molecules: List[Dict]) -> None:
        """Render results table."""
        if not molecules:
            st.warning("No molecules to display")
            return

        try:
            import pandas as pd

            df_data = []
            for i, mol in enumerate(molecules[:10], 1):
                df_data.append(
                    {
                        "Rank": i,
                        "SMILES": mol.get("smiles", "N/A")[:40] + "...",
                        "Affinity": f"{mol.get('affinity', 0):.4f}",
                        "Toxicity": f"{mol.get('toxicity', 0):.4f}",
                        "QED": f"{mol.get('qed', 0):.4f}",
                        "SA": f"{mol.get('sa', 0):.4f}",
                        "Score": f"{mol.get('composite_score', 0):.4f}",
                    }
                )

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"Could not render table: {e}")

    def render_export_options(self, results: Dict) -> None:
        """Render export options."""
        st.subheader("Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            json_data = json.dumps(results, indent=2)
            st.download_button(
                "Download Full Results (JSON)",
                json_data,
                file_name=f"bioquest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with col2:
            if results.get("best_molecule"):
                best_smiles = results["best_molecule"].get("smiles", "N/A")
                st.download_button(
                    "Best Molecule SMILES",
                    best_smiles,
                    file_name=f"best_molecule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.smi",
                    mime="text/plain",
                    use_container_width=True,
                )

        with col3:
            csv_data = "smiles,affinity,toxicity,qed,sa,composite_score\n"
            for mol in results.get("top_10", []):
                csv_data += f"{mol.get('smiles', '')},{mol.get('affinity', '')},{mol.get('toxicity', '')},{mol.get('qed', '')},{mol.get('sa', '')},{mol.get('composite_score', '')}\n"
            st.download_button(
                "Top 10 Molecules (CSV)",
                csv_data,
                file_name=f"top_molecules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    def render_results_page(self) -> None:
        """Render the results page."""
        st.header("Step 3: Analyze Results")

        if "results" not in st.session_state or not st.session_state.results:
            st.info("ℹNo results to display. Please run an optimization first.")
            return

        results = st.session_state.results

        # Best molecule section
        best = results.get("best_molecule")
        if best:
            st.subheader("Best Discovered Molecule")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Affinity", f"{best.get('affinity', 0):.4f}")
            with col2:
                st.metric("Toxicity", f"{best.get('toxicity', 0):.4f}")
            with col3:
                st.metric("QED", f"{best.get('qed', 0):.4f}")
            with col4:
                st.metric("SA", f"{best.get('sa', 0):.4f}")
            with col5:
                st.metric(
                    "Composite", f"{best.get('composite_score', 0):.4f}", delta="Best"
                )

            with st.expander("Full Details"):
                st.code(best.get("smiles", "N/A"), language="smiles")
                st.json({k: v for k, v in best.items() if k != "smiles"})

        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Convergence", "Pareto Front", "Top 10", "Statistics"]
        )

        with tab1:
            self.render_convergence_plot(results.get("convergence_metrics", {}))

        with tab2:
            self.render_pareto_front(results.get("pareto_front", []))

        with tab3:
            self.render_results_table(results.get("top_10", []))

        with tab4:
            stats = results.get("agent_statistics", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Iterations", results.get("total_iterations", 0))
            with col2:
                st.metric(
                    "Molecules Generated", results.get("total_molecules_generated", 0)
                )
            with col3:
                st.metric(
                    "Molecules Evaluated", results.get("total_molecules_evaluated", 0)
                )

            if stats:
                st.subheader("Agent Statistics")
                for agent_name, agent_stats in stats.items():
                    with st.expander(f"{agent_name}"):
                        st.json(agent_stats)

        # Export section
        st.divider()
        self.render_export_options(results)

    def run(self) -> None:
        """Main Streamlit application entry point."""
        self.render_header()
        page = self.render_sidebar()

        if page == "Configuration":
            self.render_config_page()
        elif page == "Run Optimization":
            self.render_run_page()
        elif page == "Results":
            self.render_results_page()


def main():
    """Main entry point for Streamlit app."""
    # Initialize session state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "config" not in st.session_state:
        st.session_state.config = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "page_navigation" not in st.session_state:
        st.session_state.page_navigation = "Configuration"

    # Create and run UI
    ui = BioQuestStreamlitUI()
    ui.run()


if __name__ == "__main__":
    main()
