import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from database import init_db
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Configure structured logging
logger = logging.getLogger("APIServer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_file)
init_db()

from backend_pipeline import run_risk_evaluation, preload_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing massive ML weights into RAM...")
    preload_models()
    logger.info("ML weights actively cached in High-Speed Memory.")
    yield
    logger.info("Shutting down Risk API.")

app = FastAPI(
    title="Financial Risk Manager API", 
    description="High-throughput asynchronous ML evaluation router with P99 latency protection.",
    lifespan=lifespan
)

# STRICT ORIGINS: Securing the API from external web embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5000",
        "http://127.0.0.1",
        "http://127.0.0.1:5000",
        "https://financeriskmanager.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionData(BaseModel):
    amount: float = Field(..., ge=0)
    ip_txn: int = Field(..., ge=0)
    device_txn: int = Field(..., ge=0)
    proxy_used: bool
    proxy_ip: bool
    bill_ship_mismatch: bool

class ReturnData(BaseModel):
    time_to_return_days: int
    item_margin: float
    ltv: float
    prior_returns: int
    prior_fraud: int
    category: str
    claim: str

class UIOptions(BaseModel):
    run_abuse_ring_check: bool
    generate_ai_summary: bool
    generate_chargeback: bool
    mode: str

class RiskPayload(BaseModel):
    transaction_data: TransactionData
    return_data: ReturnData
    ui_options: UIOptions
    user_id: str = "LIVE_USER"

@app.get("/")
def serve_ui():
    """Serve the frontend HTML at the root URL."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/api/evaluate")
async def evaluate_risk(payload: RiskPayload):
    """Receive a typed JSON payload from the frontend, run the async ML pipeline."""
    try:
        results = await run_risk_evaluation(payload.model_dump())
        return results
    except Exception as e:
        logger.error(f"[Server Error] Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "framework": "FastAPI", "models_cached": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=5000)