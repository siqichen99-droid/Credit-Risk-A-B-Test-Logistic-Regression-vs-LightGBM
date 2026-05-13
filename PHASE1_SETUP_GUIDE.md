# Phase 1 — FastAPI Setup Guide

## What FastAPI is and why it matters

Your LightGBM model currently lives inside a Jupyter notebook. It cannot talk to
the outside world. FastAPI solves this by wrapping your model in a **REST API** —
a web service that accepts HTTP requests (applicant data) and returns HTTP responses
(default probability + decision).

This is exactly how models work in production at companies like Fannie Mae,
Stripe, and any institution running ML in their systems.

---

## Step 1 — Reorganize your project folder

Your project folder should look like this before starting:

```
credit_risk_ab_test/
│
├── models/
│   ├── model_b_lightgbm.pkl      ← your trained LightGBM
│   ├── model_a_logistic.pkl
│   └── scaler.pkl
│
├── results/
│   └── feature_cols.txt          ← list of feature names from Section 1
│
├── main.py                       ← the FastAPI app (new file)
├── test_api.py                   ← the test script (new file)
└── requirements.txt              ← package list (new file)
```

Copy `main.py`, `test_api.py`, and `requirements.txt` into the root of your
`credit_risk_ab_test` folder.

---

## Step 2 — Install FastAPI packages

Open your terminal, navigate to your project folder, activate your virtual
environment, then run:

```bash
# Navigate to your project folder
cd "C:\Users\hanto\OneDrive\Desktop\Siqi Chen Projects\ML\credit_risk_ab_test"

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install FastAPI and its server (uvicorn)
pip install fastapi uvicorn pydantic requests
```

You should see packages downloading. This takes about 1 minute.

**What each package does:**
- `fastapi` — the framework that turns Python functions into API endpoints
- `uvicorn` — the server that runs FastAPI (like a mini web server on your laptop)
- `pydantic` — validates that incoming data has the right types and fields
- `requests` — used by the test script to send HTTP requests to the API

---

## Step 3 — Start the API server

In your terminal (with virtual environment active), run:

```bash
uvicorn main:app --reload
```

**What this command means:**
- `main` = the filename (`main.py`)
- `app` = the FastAPI object inside that file (`app = FastAPI(...)`)
- `--reload` = automatically restart when you save changes (development mode)

You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Application startup complete.
```

The API is now running on your machine at `http://127.0.0.1:8000`.
Do NOT close this terminal — it keeps the server alive.

---

## Step 4 — Explore the automatic documentation

FastAPI automatically generates interactive API documentation.

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

You will see the **Swagger UI** — a full documentation page listing all your
endpoints. You can:
- Click on `/predict`
- Click "Try it out"
- Fill in applicant details
- Click "Execute"
- See the prediction response in real time

This is the page you will show interviewers during interviews.

---

## Step 5 — Run the test script

Open a **second terminal** (keep the first one running the server).
Navigate to your project folder, activate venv, then run:

```bash
python test_api.py
```

You should see output like:
```
Credit Risk API — Test Suite
==================================================

[1] Health check
    Status: healthy
    Model loaded: True
    Features: 26

==================================================
  Low risk applicant — expect: Approve
==================================================
  Default probability: 0.0423
  Decision:            Approve
  Risk tier:           Very Low
  Threshold used:      0.168

==================================================
  High risk applicant — expect: Decline
==================================================
  Default probability: 0.4821
  Decision:            Decline
  Risk tier:           High
  Threshold used:      0.168
```

If you see this output — Phase 1 is complete.

---

## What just happened (the full picture)

```
test_api.py                    main.py                        model
    │                              │                              │
    │  POST /predict                │                              │
    │  {AMT_INCOME: 135000,  ──────►│  1. Validate input           │
    │   AMT_CREDIT: 450000,         │     (pydantic)               │
    │   EXT_SOURCE_2: 0.59,         │                              │
    │   ...}                        │  2. Engineer features         │
    │                              │     (DTI, LTV, age...)        │
    │                              │                              │
    │                              │  3. model.predict_proba() ──►│
    │                              │                              │
    │                              │◄── probability = 0.1823 ─────│
    │                              │
    │                              │  4. Apply threshold (0.168)
    │                              │     → Decision: Decline
    │                              │
    │◄── {probability: 0.1823,      │
    │     decision: "Decline",      │
    │     risk_tier: "Medium"}      │
```

Every time someone sends applicant data, the API:
1. Validates the input fields (pydantic)
2. Engineers the 9 financial features (same as Section 1)
3. Runs the LightGBM model
4. Applies the optimal threshold (0.168) from Section 3
5. Returns the probability, decision, and risk tier

---

## Common errors and fixes

**Error: `ModuleNotFoundError: No module named 'fastapi'`**
→ Your virtual environment is not active. Run `venv\Scripts\activate` first.

**Error: `FileNotFoundError: models/model_b_lightgbm.pkl`**
→ The `models/` folder is not in the right place. Make sure it sits alongside
`main.py` in the same directory.

**Error: `Address already in use`**
→ Another process is using port 8000. Stop it with CTRL+C, or run on a
different port: `uvicorn main:app --reload --port 8001`

**Error: `feature_cols.txt not found`**
→ Make sure `feature_cols.txt` is inside the `results/` folder.
It was generated in Section 1 Cell 11.

---

## Phase 1 complete checklist

- [ ] `main.py` and `test_api.py` placed in project root
- [ ] FastAPI packages installed (`pip install fastapi uvicorn pydantic requests`)
- [ ] Server starts with `uvicorn main:app --reload`
- [ ] http://127.0.0.1:8000/docs opens in browser
- [ ] `python test_api.py` runs without errors
- [ ] Low risk applicant returns Approve
- [ ] High risk applicant returns Decline

**Next: Phase 2 — MLflow experiment tracking**
