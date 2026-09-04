# Financial Risk Command Center

A production-grade, end-to-end financial risk management system built with a **cascading multi-model ML architecture**. The platform evaluates checkout fraud, coordinated abuse rings, and post-purchase return abuse in real time, with an integrated generative AI layer for forensic reporting and legal dispute generation.

> **Live Demo:** [financeriskmanager.onrender.com](https://financeriskmanager.onrender.com)

---

## System Architecture

![End-to-End Architecture](architecture.png)

The system is organized into three operational phases, each backed by dedicated ML models and decision routers.

### Phase 1 — Checkout & Network Fraud (Model 1 → Model 2)

![Fraud Spike & Abuse Ring Flow](phase1_flow.png)

A **cascading two-model pipeline** where every transaction flows through Model 1 (Fraud Spike), and then all outcomes — including approvals — are routed through Model 2 (Abuse Ring) for secondary network-level verification.

| Model | Type | Purpose |
|---|---|---|
| **Model 1** — Fraud Spike | LightGBM (Focal Loss) | Evaluates transaction amount, velocity, and engineered features to produce a fraud probability via sigmoid-transformed log-odds |
| **Model 2** — Abuse Ring Sentinel | PyTorch Autoencoder | Detects coordinated fraud networks by measuring reconstruction error (MSE) on transaction telemetry, fused with categorical proxy/mismatch signals |

**Decision Flow:**
- **Model 1 → Approve / Uncertain / Fraud** based on calibrated F1 thresholds
- **All three outcomes** are forwarded to Model 2 for abuse ring screening:
  - `Approve` + Ring Detected → `MANUAL REVIEW (ABUSE RING)` *(catches low-and-slow carding attacks)*
  - `Fraud` + Ring Detected → `AUTO-BLOCK (ABUSE RING)` *(forensic tagging for dispute evidence)*
  - `Uncertain` path → future implementation for priority manual review

### Phase 2 — Post-Purchase: Returns & Policy Abuse (Model 3)

![Return Risk Flow](phase2_flow.png)

| Model | Type | Purpose |
|---|---|---|
| **Model 3** — Behavioral Scorer | LightGBM (Binary) | Scores return requests using customer LTV, prior return history, claim type, item margin, and time-to-return |

**Tri-State Financial Router:**
- `p < t_low` → **Auto-Refund**
- `t_low ≤ p ≤ t_high` → **Require Photo**
- `p > t_high` → **Auto-Reject**

Thresholds (`t_low`, `t_high`) are derived from a vectorized cost-landscape grid search that optimizes for net financial impact — not just F1 score.

### Phase 3 — Dispute Resolution (Generative AI)

![Chargeback Generation Flow](phase3_flow.png)

When fraud is flagged, an **async LLM engine** (Groq) generates:
- **Forensic Summary** — 3-sentence analytical breakdown of ML signals
- **Chargeback Dispute Letter** — Formal legal letter addressed to Visa/Mastercard adjudication teams, citing specific fraud evidence

Both outputs are generated in parallel via `asyncio.gather()` using LangChain prompt templates.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **ML Inference** | LightGBM, PyTorch, scikit-learn |
| **Generative AI** | Groq (LangChain), async prompt chains |
| **Database** | SQLite (Unified User Database) |
| **Frontend** | Vanilla HTML/CSS/JS (glassmorphism UI) |
| **Deployment** | Docker, Render |

---

## Project Structure

```
FinanceRiskManagerAI/
├── server.py                    # FastAPI application with Pydantic validation
├── backend_pipeline.py          # Cascading ML evaluation engine (Tracks 1-5)
├── ai_service.py                # Async LLM service (Groq/LangChain)
├── database.py                  # SQLite unified user database
├── index.html                   # Frontend Command Center UI
├── validation.py                # Data integrity & leakage checks
│
├── dataA(opensource).py          # IEEE dataset fetcher + EDA
├── dataB(generated).py          # Causal feature engineering + synthetic data
├── generate_returns_data.py     # Persona-based return dataset generator
├── calibrate_return_thresholds.py  # Financial cost-landscape grid search
│
├── phase1.ipynb                 # EDA & baseline modeling
├── phase2.ipynb                 # Feature engineering deep-dive
├── phase3.ipynb                 # Model training & evaluation
├── phase4_*.ipynb               # Final evaluation & review reduction
├── train_return_scorer.ipynb    # Return risk model training
│
├── *.pkl / *.pth                # Trained model artifacts
├── Dockerfile                   # Production container config
├── requirements.txt             # Pinned dependencies (CPU PyTorch)
└── .dockerignore                # Keeps Docker image lean
```

---

## ML Pipeline Details

### Data Engineering
- **Source:** IEEE-CIS Fraud Detection dataset (590K+ transactions)
- **Causal Feature Engineering:** Rolling velocity counts computed using `np.searchsorted` to prevent temporal leakage — features are strictly derived from past transactions only
- **Validation:** Automated checks for temporal boundary integrity, causal leakage, velocity ordering consistency, and feature distribution drift

### Model Training
- **Fraud Spike (Model 1):** LightGBM trained with focal loss on temporally-split data (M1-M4 train, M5 calibration, M6 test). Heuristic log-odds adjustments are applied at inference to amplify real-time velocity/proxy signals that couldn't be natively trained into the IEEE subset
- **Abuse Ring Sentinel (Model 2):** PyTorch autoencoder (128→64→16→8 latent) with LayerNorm, GELU activations, and dropout. Anomaly detection via reconstruction MSE threshold, fused with categorical proxy/mismatch indicators
- **Return Behavioral Scorer (Model 3):** LightGBM binary classifier on persona-based synthetic return data. Thresholds calibrated via financial cost-landscape optimization (photo review cost, friction penalty, LTV loss modeling)

### Closed-Loop Architecture
All ML decisions are logged back to the **Unified User Database** (SQLite), updating each user's LTV, return count, and fraud tags. Future evaluations read from this state, creating a feedback loop where historical behavior influences real-time risk scoring.

---

## Getting Started

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/) (for AI features)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/SakethKakarla14/FinanceRiskManagerAI.git
cd FinanceRiskManagerAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GROQ_API_KEY=your_key_here" > .env

# Run the server
python run_server.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Docker Deployment

```bash
docker build -t finance-risk-manager .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here finance-risk-manager
```

---

## Demo Walkthrough

The UI provides a **3-step wizard flow**:

1. **Configure Evaluation Tracks** — Toggle Fraud Spike, Abuse Ring, AI Summary, and Chargeback generation
2. **Enter Telemetry Data** — Use preset scenarios (Safe Purchase, High-Speed Spike, Proxy Abuse Ring, etc.) or enter custom values across Transaction and Return tabs
3. **View Analysis Output** — Color-coded ML decisions with optional AI-generated forensic reports

### Sample Scenarios

| Scenario | Expected Fraud Spike | Expected Abuse Ring | Expected Return Risk |
|---|---|---|---|
| Safe Purchase ($45, velocity=1) | AUTO-APPROVE | CLEARED | — |
| High-Speed Spike ($3200, velocity=12) | AUTO-BLOCK | CLEARED | — |
| Proxy Abuse Ring ($150, proxy+mismatch) | AUTO-BLOCK | DETECTED | — |
| Serial Wardrober (29 days, 3 returns) | — | — | REQUIRE PHOTO |
| Empty Box Scam (prior fraud=1) | — | — | AUTO-REJECT |

---

## Key Design Decisions

1. **Why Model 2 runs on AUTO-APPROVE transactions:** Sophisticated attackers use "low-and-slow" carding attacks — small purchases ($5-$15) that bypass volume-based fraud models. Model 2 catches these by analyzing network infrastructure (proxies, IP mismatches) rather than transaction characteristics.

2. **Why heuristic log-odds boosters exist:** The IEEE dataset doesn't contain real-time velocity or proxy features. At inference time, these signals are injected as calibrated log-odds adjustments to the LightGBM leaf output before sigmoid transformation. This is explicitly documented as a heuristic, not a learned parameter.

3. **Why financial thresholds, not F1:** The return risk router uses thresholds derived from a dollar-denominated cost landscape, not accuracy metrics. This ensures the system optimizes for net business impact (accounting for photo review costs, customer friction, and LTV loss).

---

## License

This project is for educational and portfolio purposes.
