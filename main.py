"""
Credit Risk Model API
FastAPI application that serves the LightGBM credit default prediction model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts the probability of loan default using a LightGBM model trained on the Home Credit dataset.",
    version="1.0.0"
)

# ── Load model and artifacts on startup ──────────────────────────────────────

BASE = Path(__file__).parent

model    = joblib.load(BASE / "models" / "model_b_lightgbm.pkl")
scaler   = joblib.load(BASE / "models" / "scaler.pkl")

with open(BASE / "results" / "feature_cols.txt") as f:
    FEATURE_COLS = f.read().splitlines()


# ── Request schema ────────────────────────────────────────────────────────────
# These are the raw input fields an applicant provides.
# Pydantic validates every field automatically — wrong types return a 422 error.

class ApplicantInput(BaseModel):
    AMT_INCOME_TOTAL:    float = Field(..., example=135000.0,  description="Annual income ($)")
    AMT_CREDIT:          float = Field(..., example=450000.0,  description="Loan amount requested ($)")
    AMT_ANNUITY:         float = Field(..., example=22500.0,   description="Annual loan repayment ($)")
    AMT_GOODS_PRICE:     float = Field(..., example=400000.0,  description="Price of goods financed ($)")
    DAYS_BIRTH:          float = Field(..., example=-12000.0,  description="Age in days (negative)")
    DAYS_EMPLOYED:       float = Field(..., example=-2000.0,   description="Employment duration in days (negative if employed)")
    DAYS_ID_PUBLISH:     float = Field(..., example=-1500.0,   description="Days since ID was last changed (negative)")
    DAYS_REGISTRATION:   float = Field(..., example=-3000.0,   description="Days since registration change (negative)")
    EXT_SOURCE_1:        float = Field(..., example=0.51,      description="External credit score 1 (0–1)")
    EXT_SOURCE_2:        float = Field(..., example=0.59,      description="External credit score 2 (0–1)")
    EXT_SOURCE_3:        float = Field(..., example=0.49,      description="External credit score 3 (0–1)")
    NAME_CONTRACT_TYPE:  str   = Field(..., example="Cash loans",   description="Cash loans or Revolving loans")
    CODE_GENDER:         str   = Field(..., example="M",            description="M or F")
    FLAG_OWN_CAR:        str   = Field(..., example="N",            description="Y or N")
    FLAG_OWN_REALTY:     str   = Field(..., example="Y",            description="Y or N")
    CNT_CHILDREN:        int   = Field(..., example=0,              description="Number of children")
    NAME_INCOME_TYPE:    str   = Field(..., example="Working",      description="Income type")
    NAME_EDUCATION_TYPE: str   = Field(..., example="Secondary / secondary special", description="Education level")
    NAME_FAMILY_STATUS:  str   = Field(..., example="Married",      description="Marital status")
    REGION_RATING_CLIENT:int   = Field(..., example=2,              description="Region risk rating (1=low, 3=high)")


# ── Feature engineering ───────────────────────────────────────────────────────
# Replicates exactly the transformations from Section 1 notebook.

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Ratio features
    df["DEBT_TO_INCOME"]     = df["AMT_CREDIT"]      / (df["AMT_INCOME_TOTAL"] + 1)
    df["PAYMENT_TO_INCOME"]  = df["AMT_ANNUITY"]      / (df["AMT_INCOME_TOTAL"] + 1)
    df["LOAN_TO_VALUE"]      = df["AMT_CREDIT"]       / (df["AMT_GOODS_PRICE"]  + 1)
    df["CREDIT_TERM"]        = df["AMT_CREDIT"]       / (df["AMT_ANNUITY"]      + 1)

    # Age and employment
    df["AGE_YEARS"]          = (-df["DAYS_BIRTH"])    / 365
    emp = df["DAYS_EMPLOYED"].replace(365243, 0)
    df["EMPLOYED_YEARS"]     = (-emp)                 / 365
    df["EMPLOYED_YEARS"]     = df["EMPLOYED_YEARS"].clip(lower=0)
    df["EMPLOYMENT_TO_AGE"]  = df["EMPLOYED_YEARS"]   / (df["AGE_YEARS"] + 1)

    # Bureau score aggregates
    df["EXT_SOURCE_MEAN"]    = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_MIN"]     = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    # Drop raw day columns (replaced by engineered versions)
    drop_cols = ["DAYS_BIRTH", "DAYS_EMPLOYED"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Align to training feature set — fill any missing with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLS]


# ── Response schema ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Predicted probability of default (0–1)")
    decision:            str   = Field(..., description="Approve or Decline based on optimal threshold")
    risk_tier:           str   = Field(..., description="Very Low / Low / Medium / High / Very High")
    threshold_used:      float = Field(..., description="Decision threshold applied")
    model_version:       str   = Field(..., description="Model identifier")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {
        "status":      "online",
        "api":         "Credit Risk Prediction API",
        "version":     "1.0.0",
        "description": "POST applicant data to /predict to receive a default probability."
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check including model status."""
    return {
        "status":         "healthy",
        "model_loaded":   model is not None,
        "model_type":     type(model).__name__,
        "features_count": len(FEATURE_COLS)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(applicant: ApplicantInput):
    """
    Predict the probability of loan default for a single applicant.

    - Returns a probability score between 0 and 1
    - Applies the optimal threshold (0.168) found in Section 3
    - Returns a risk tier classification and approve/decline decision
    """
    try:
        # Build feature dataframe
        features = engineer_features(applicant.dict())

        # Predict probability (column 1 = P(default))
        prob = float(model.predict_proba(features)[0][1])

        # Apply optimal threshold from Section 3
        THRESHOLD = 0.168
        decision  = "Decline" if prob >= THRESHOLD else "Approve"

        # Risk tier classification
        if prob < 0.10:
            tier = "Very Low"
        elif prob < 0.20:
            tier = "Low"
        elif prob < 0.35:
            tier = "Medium"
        elif prob < 0.50:
            tier = "High"
        else:
            tier = "Very High"

        return PredictionResponse(
            default_probability = round(prob, 4),
            decision            = decision,
            risk_tier           = tier,
            threshold_used      = THRESHOLD,
            model_version       = "lightgbm-v1.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(applicants: list[ApplicantInput]):
    """
    Predict default probability for multiple applicants in one request.
    Maximum 100 applicants per batch.
    """
    if len(applicants) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 100 applicants."
        )

    results = []
    THRESHOLD = 0.168

    for i, applicant in enumerate(applicants):
        try:
            features = engineer_features(applicant.dict())
            prob     = float(model.predict_proba(features)[0][1])
            decision = "Decline" if prob >= THRESHOLD else "Approve"
            results.append({
                "applicant_index":    i,
                "default_probability": round(prob, 4),
                "decision":           decision,
            })
        except Exception as e:
            results.append({
                "applicant_index": i,
                "error":           str(e)
            })

    return {
        "total":    len(results),
        "approved": sum(1 for r in results if r.get("decision") == "Approve"),
        "declined": sum(1 for r in results if r.get("decision") == "Decline"),
        "results":  results
    }
