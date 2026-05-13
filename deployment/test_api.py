"""
API Test Script
Run this after starting the FastAPI server to verify all endpoints work correctly.
Usage: python test_api.py
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# ── Sample applicants ──────────────────────────────────────────────────────────
# These represent three different risk profiles to test the API.

LOW_RISK_APPLICANT = {
    "AMT_INCOME_TOTAL":     270000.0,
    "AMT_CREDIT":           450000.0,
    "AMT_ANNUITY":          22500.0,
    "AMT_GOODS_PRICE":      450000.0,
    "DAYS_BIRTH":           -16000.0,
    "DAYS_EMPLOYED":        -3650.0,
    "DAYS_ID_PUBLISH":      -1000.0,
    "DAYS_REGISTRATION":    -5000.0,
    "EXT_SOURCE_1":         0.72,
    "EXT_SOURCE_2":         0.80,
    "EXT_SOURCE_3":         0.75,
    "NAME_CONTRACT_TYPE":   "Cash loans",
    "CODE_GENDER":          "F",
    "FLAG_OWN_CAR":         "Y",
    "FLAG_OWN_REALTY":      "Y",
    "CNT_CHILDREN":         0,
    "NAME_INCOME_TYPE":     "Working",
    "NAME_EDUCATION_TYPE":  "Higher education",
    "NAME_FAMILY_STATUS":   "Married",
    "REGION_RATING_CLIENT": 1
}

MEDIUM_RISK_APPLICANT = {
    "AMT_INCOME_TOTAL":     135000.0,
    "AMT_CREDIT":           450000.0,
    "AMT_ANNUITY":          22500.0,
    "AMT_GOODS_PRICE":      400000.0,
    "DAYS_BIRTH":           -12000.0,
    "DAYS_EMPLOYED":        -2000.0,
    "DAYS_ID_PUBLISH":      -1500.0,
    "DAYS_REGISTRATION":    -3000.0,
    "EXT_SOURCE_1":         0.51,
    "EXT_SOURCE_2":         0.59,
    "EXT_SOURCE_3":         0.49,
    "NAME_CONTRACT_TYPE":   "Cash loans",
    "CODE_GENDER":          "M",
    "FLAG_OWN_CAR":         "N",
    "FLAG_OWN_REALTY":      "Y",
    "CNT_CHILDREN":         0,
    "NAME_INCOME_TYPE":     "Working",
    "NAME_EDUCATION_TYPE":  "Secondary / secondary special",
    "NAME_FAMILY_STATUS":   "Married",
    "REGION_RATING_CLIENT": 2
}

HIGH_RISK_APPLICANT = {
    "AMT_INCOME_TOTAL":     45000.0,
    "AMT_CREDIT":           360000.0,
    "AMT_ANNUITY":          36000.0,
    "AMT_GOODS_PRICE":      300000.0,
    "DAYS_BIRTH":           -9000.0,
    "DAYS_EMPLOYED":        -180.0,
    "DAYS_ID_PUBLISH":      -300.0,
    "DAYS_REGISTRATION":    -500.0,
    "EXT_SOURCE_1":         0.18,
    "EXT_SOURCE_2":         0.22,
    "EXT_SOURCE_3":         0.15,
    "NAME_CONTRACT_TYPE":   "Cash loans",
    "CODE_GENDER":          "M",
    "FLAG_OWN_CAR":         "N",
    "FLAG_OWN_REALTY":      "N",
    "CNT_CHILDREN":         2,
    "NAME_INCOME_TYPE":     "Working",
    "NAME_EDUCATION_TYPE":  "Lower secondary",
    "NAME_FAMILY_STATUS":   "Single / not married",
    "REGION_RATING_CLIENT": 3
}


def print_result(label, response):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Default probability: {data['default_probability']:.4f}")
        print(f"  Decision:            {data['decision']}")
        print(f"  Risk tier:           {data['risk_tier']}")
        print(f"  Threshold used:      {data['threshold_used']}")
    else:
        print(f"  ERROR {response.status_code}: {response.text}")


def run_tests():
    print("\nCredit Risk API — Test Suite")
    print("="*50)

    # Test 1: Health check
    print("\n[1] Health check")
    r = requests.get(f"{BASE_URL}/health")
    print(f"    Status: {r.json()['status']}")
    print(f"    Model loaded: {r.json()['model_loaded']}")
    print(f"    Features: {r.json()['features_count']}")

    # Test 2: Low risk applicant
    r = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_APPLICANT)
    print_result("Low risk applicant — expect: Approve", r)

    # Test 3: Medium risk applicant
    r = requests.post(f"{BASE_URL}/predict", json=MEDIUM_RISK_APPLICANT)
    print_result("Medium risk applicant — expect: borderline", r)

    # Test 4: High risk applicant
    r = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_APPLICANT)
    print_result("High risk applicant — expect: Decline", r)

    # Test 5: Batch prediction
    print(f"\n{'='*50}")
    print("  Batch prediction (3 applicants)")
    print(f"{'='*50}")
    r = requests.post(
        f"{BASE_URL}/predict/batch",
        json=[LOW_RISK_APPLICANT, MEDIUM_RISK_APPLICANT, HIGH_RISK_APPLICANT]
    )
    if r.status_code == 200:
        data = r.json()
        print(f"  Total:    {data['total']}")
        print(f"  Approved: {data['approved']}")
        print(f"  Declined: {data['declined']}")
        for res in data["results"]:
            print(f"  Applicant {res['applicant_index']}: "
                  f"prob={res['default_probability']:.4f} → {res['decision']}")

    print("\n\nAll tests complete.")
    print("Open http://127.0.0.1:8000/docs for the interactive API documentation.")


if __name__ == "__main__":
    run_tests()
