#!/usr/bin/env python3
"""
Verify that predictions are validation-only (each case appears exactly once per seed).
"""

import pandas as pd

preds = pd.read_csv('legalbert_multiseed_attention/all_predictions.csv')

print("="*60)
print("VALIDATION-ONLY CHECK")
print("="*60)

# Check: does each case appear exactly once per seed?
check = preds.groupby(['seed', 'case_id'])['fold'].nunique()

multi_fold = (check > 1).sum()
single_fold = (check == 1).sum()

print(f"\nCases appearing in >1 fold per seed: {multi_fold}")
print(f"Cases appearing in exactly 1 fold per seed: {single_fold}")

print(f"\nTotal predictions: {len(preds)}")
print(f"Expected (1961 cases × 5 seeds): {1961 * 5}")

print("\n" + "="*60)
if multi_fold == 0 and single_fold == 9805:
    print("✅ VERIFIED: Each case appears exactly once per seed")
    print("   → Fold distribution claim is VALID")
else:
    print("⚠️ WARNING: Predictions may not be validation-only")
    print("   → Fold distribution claim needs hedging")
print("="*60)
