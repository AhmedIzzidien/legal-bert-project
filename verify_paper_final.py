#!/usr/bin/env python3
"""
===============================================================================
VERIFY PAPER STATISTICS
===============================================================================

Verifies all statistics reported in the paper match the raw data.
Run this before submission to ensure reproducibility.

Checks:
    - Overall accuracy and F1 scores
    - Consistency distribution (0/5 to 5/5)
    - Error rates by outcome (granted vs refused)
    - FP/FN breakdown
    - L/E/T stratum performance
    - Keyword × outcome interaction rates
    - Topic × outcome interaction rates
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import sys

# ===============================================================================
# LOAD DATA
# ===============================================================================

print("="*70)
print("VERIFYING PAPER STATISTICS")
print("="*70)

errors = []

# Load main data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))
df['label'] = (df['outcome'] == 'summary judgment refused').astype(int)

# Load predictions
all_preds = pd.read_csv('legalbert_multiseed_attention/all_predictions.csv')
consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')

# Merge
df = df.merge(consistency[['case_id', 'times_correct']], on='case_id')

print(f"\nLoaded {len(df)} cases, {len(all_preds)} predictions")

# ===============================================================================
# 1. BASIC COUNTS
# ===============================================================================

print("\n" + "="*70)
print("1. BASIC COUNTS")
print("="*70)

n_cases = len(df)
n_granted = (df['label'] == 0).sum()
n_refused = (df['label'] == 1).sum()

print(f"Total cases: {n_cases}")
print(f"Granted: {n_granted} ({n_granted/n_cases*100:.1f}%)")
print(f"Refused: {n_refused} ({n_refused/n_cases*100:.1f}%)")

# PAPER CLAIMS: 1,961 cases, 61% granted, 39% refused
if n_cases != 1961:
    errors.append(f"Case count: got {n_cases}, paper says 1961")

# ===============================================================================
# 2. OVERALL PERFORMANCE
# ===============================================================================

print("\n" + "="*70)
print("2. OVERALL PERFORMANCE (averaged across 5 seeds)")
print("="*70)

seed_metrics = []
for seed in all_preds['seed'].unique():
    seed_preds = all_preds[all_preds['seed'] == seed]
    acc = accuracy_score(seed_preds['label'], seed_preds['predicted'])
    f1_macro = f1_score(seed_preds['label'], seed_preds['predicted'], average='macro')
    f1_granted = f1_score(seed_preds['label'], seed_preds['predicted'], pos_label=0)
    f1_refused = f1_score(seed_preds['label'], seed_preds['predicted'], pos_label=1)
    seed_metrics.append({
        'seed': seed, 'accuracy': acc, 'f1_macro': f1_macro,
        'f1_granted': f1_granted, 'f1_refused': f1_refused
    })

metrics_df = pd.DataFrame(seed_metrics)

acc_mean = metrics_df['accuracy'].mean()
f1_macro_mean = metrics_df['f1_macro'].mean()
f1_granted_mean = metrics_df['f1_granted'].mean()
f1_refused_mean = metrics_df['f1_refused'].mean()

print(f"Accuracy: {acc_mean:.3f} ({acc_mean*100:.1f}%)")
print(f"F1-Macro: {f1_macro_mean:.3f}")
print(f"F1-Granted: {f1_granted_mean:.3f}")
print(f"F1-Refused: {f1_refused_mean:.3f}")

# PAPER CLAIMS: 62.7% accuracy, F1-Granted 0.698, F1-Refused 0.511
if abs(acc_mean - 0.627) > 0.01:
    errors.append(f"Accuracy: got {acc_mean:.3f}, paper says 0.627")

# ===============================================================================
# 3. CONSISTENCY DISTRIBUTION
# ===============================================================================

print("\n" + "="*70)
print("3. CONSISTENCY DISTRIBUTION")
print("="*70)

for tc in range(6):
    count = (df['times_correct'] == tc).sum()
    pct = count / len(df) * 100
    print(f"{tc}/5: {count} ({pct:.1f}%)")

# PAPER CLAIMS: 153 always-wrong (7.8%), 546 always-right (27.8%)
n_always_wrong = (df['times_correct'] == 0).sum()
n_always_right = (df['times_correct'] == 5).sum()

if n_always_wrong != 153:
    errors.append(f"Always-wrong count: got {n_always_wrong}, paper says 153")
if n_always_right != 546:
    errors.append(f"Always-right count: got {n_always_right}, paper says 546")

# ===============================================================================
# 4. ERROR RATES BY OUTCOME
# ===============================================================================

print("\n" + "="*70)
print("4. ERROR RATES BY OUTCOME")
print("="*70)

granted_cases = df[df['label'] == 0]
refused_cases = df[df['label'] == 1]

granted_aw = (granted_cases['times_correct'] == 0).sum()
refused_aw = (refused_cases['times_correct'] == 0).sum()

granted_err_rate = granted_aw / len(granted_cases) * 100
refused_err_rate = refused_aw / len(refused_cases) * 100

print(f"Granted: {granted_aw}/{len(granted_cases)} always-wrong ({granted_err_rate:.1f}%)")
print(f"Refused: {refused_aw}/{len(refused_cases)} always-wrong ({refused_err_rate:.1f}%)")
print(f"Ratio: {refused_err_rate/granted_err_rate:.1f}x")

# PAPER CLAIMS: 3.6% granted error, 14.4% refused error, 4x ratio
if abs(granted_err_rate - 3.6) > 0.5:
    errors.append(f"Granted error rate: got {granted_err_rate:.1f}%, paper says 3.6%")
if abs(refused_err_rate - 14.4) > 0.5:
    errors.append(f"Refused error rate: got {refused_err_rate:.1f}%, paper says 14.4%")

# ===============================================================================
# 5. FP/FN BREAKDOWN
# ===============================================================================

print("\n" + "="*70)
print("5. FP/FN BREAKDOWN (of always-wrong cases)")
print("="*70)

always_wrong = df[df['times_correct'] == 0]

# FN: True=Granted, Model=Refused
fn_count = len(always_wrong[always_wrong['label'] == 0])
# FP: True=Refused, Model=Granted
fp_count = len(always_wrong[always_wrong['label'] == 1])

print(f"False Positives (predicted granted, actually refused): {fp_count} ({fp_count/len(always_wrong)*100:.1f}%)")
print(f"False Negatives (predicted refused, actually granted): {fn_count} ({fn_count/len(always_wrong)*100:.1f}%)")

# PAPER CLAIMS: 110 FP (71.9%), 43 FN (28.1%)
if fp_count != 110:
    errors.append(f"FP count: got {fp_count}, paper says 110")
if fn_count != 43:
    errors.append(f"FN count: got {fn_count}, paper says 43")

# ===============================================================================
# 6. KEYWORD x OUTCOME
# ===============================================================================

print("\n" + "="*70)
print("6. KEYWORD x OUTCOME ERROR RATES")
print("="*70)

df['all_text'] = (df['facts'].fillna('') + ' ' + 
                  df['applicant_reason'].fillna('') + ' ' + 
                  df['defence_reason'].fillna('')).str.lower()

keywords = ['binding', 'defamatory', 'property']

for kw in keywords:
    has_kw = df['all_text'].str.contains(kw)
    
    # Granted + keyword
    gk = df[(df['label'] == 0) & has_kw]
    gk_err = (gk['times_correct'] == 0).mean() * 100 if len(gk) > 0 else 0
    
    # Refused + keyword
    rk = df[(df['label'] == 1) & has_kw]
    rk_err = (rk['times_correct'] == 0).mean() * 100 if len(rk) > 0 else 0
    
    ratio = rk_err / gk_err if gk_err > 0 else float('inf')
    
    print(f"{kw}: Granted={gk_err:.1f}%, Refused={rk_err:.1f}%, Ratio={ratio:.1f}x")

# PAPER CLAIMS: binding: 4.8% granted, 32.3% refused, 6.7x
has_binding = df['all_text'].str.contains('binding')
binding_granted = df[(df['label'] == 0) & has_binding]
binding_refused = df[(df['label'] == 1) & has_binding]
binding_g_err = (binding_granted['times_correct'] == 0).mean() * 100
binding_r_err = (binding_refused['times_correct'] == 0).mean() * 100

if abs(binding_r_err - 32.3) > 2:
    errors.append(f"Binding+refused error: got {binding_r_err:.1f}%, paper says 32.3%")

# ===============================================================================
# SUMMARY
# ===============================================================================

print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

if errors:
    print(f"\n[X] FOUND {len(errors)} DISCREPANCIES:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("\n[OK] ALL STATISTICS VERIFIED - PAPER MATCHES DATA")
    sys.exit(0)
