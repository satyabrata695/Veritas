"""Simple health-check serverless function."""
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_backed": False,
        "detector_method": "laplacian_edge_heuristic_v1",
        "warning": (
            "This Vercel serverless endpoint uses a heuristic fallback, not the "
            "trained VERITAS model. Configure the frontend to use the deployed "
            "model backend for trained-model results."
        ),
    }
