"""
FastAPI application for BioQuest.

API endpoints for molecule prediction and optimization.
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.schemas import OptimizeRequest, OptimizeResponse
from src.inference import MoleculePredictor, ModelNotLoadedError
from src.core.optimization import OptimizationEvaluator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BioQuest API",
    description="AI-Driven Drug Discovery API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None
_evaluator = None


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "BioQuest API", "version": "1.0.0"}


def get_predictor() -> MoleculePredictor:
    """Get or create molecule predictor."""
    global _predictor
    if _predictor is None:
        logger.info("Initializing MoleculePredictor")
        try:
            _predictor = MoleculePredictor(
                protein_sequence="PEPTIDE" * 10,
                use_gpu=False,
                models_dir="artifacts/models",
            )
        except ModelNotLoadedError as e:
            logger.error(f"Failed to load models: {e}")
            raise HTTPException(status_code=503, detail=str(e))
    return _predictor


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    """
    Run molecule optimization.

    Args:
        request: Optimization request parameters

    Returns:
        Optimization results with best molecules
    """
    start_time = time.time()

    try:
        predictor = get_predictor()
        evaluator = OptimizationEvaluator(
            objective_weights=request.objectives,
            plateau_threshold=0.001,
            patience=20,
        )

        batch_size = request.batch_size
        seeds = request.seeds[:batch_size]

        batch_results = predictor.batch_predict(seeds)
        population = []
        for i, smiles in enumerate(seeds):
            mol_data = {
                "smiles": smiles,
                "affinity": float(batch_results["affinity"][i]),
                "toxicity": float(batch_results["toxicity"][i]),
                "qed": float(batch_results["qed"][i]),
                "sa": float(batch_results["sa"][i]),
                "logp": float(batch_results["logp"][i]),
                "mw": float(batch_results["mw"][i]),
            }
            population.append(mol_data)

        scores = evaluator.evaluate_population(population, iteration=0)
        evaluator.update_iteration(scores, iteration=0)

        best = evaluator.get_best_molecule()
        top_5 = evaluator.get_top_molecules(k=5)
        pareto = evaluator.get_pareto_front()

        elapsed = time.time() - start_time

        return OptimizeResponse(
            best_molecule=best.to_dict() if best else None,
            top_5=[m.to_dict() for m in top_5],
            pareto_front=[m.to_dict() for m in pareto],
            total_iterations=1,
            execution_time_seconds=elapsed,
        )

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(smiles: str):
    """
    Predict properties for single molecule.

    Args:
        smiles: SMILES string

    Returns:
        Predicted properties
    """
    try:
        predictor = get_predictor()
        properties = predictor.predict_all_properties(smiles)
        return {"smiles": smiles, "properties": properties}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))