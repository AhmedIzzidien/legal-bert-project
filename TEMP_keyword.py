#!/usr/bin/env python3
"""
Compute multi-seed averaged F1 per L/E/T stratum.
Performance averaged over 5 independent random seeds.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*110)
print("MULTI-SEED L/E/T STRATUM PERFORMANCE ANALYSIS")
print("Performance averaged over 5 independent random seeds")
print("="*110)

# Load predictions
preds = pd.read_csv("legalbert_multiseed_attention/all_predictions.csv")
print(f"\nLoaded {len(preds)} predictions")
print(f"Seeds: {sorted(preds['seed'].unique())}")

# Load consistency (has stratum)
consistency = pd.read_csv("legalbert_multiseed_attention/case_consistency.csv")
print(f"Loaded {len(consistency)} cases from consistency file")

# Stratum counts
print("\nStratum distribution:")
stratum_counts = consistency['stratum'].value_counts().sort_index()
for stratum, count in stratum_counts.items():
    print(f"  {stratum}: {count}")

# Merge stratum into predictions
preds = preds.merge(consistency[['case_id', 'stratum']], on='case_id', how='left')

# =============================================================================
# COMPUTE METRICS PER STRATUM PER SEED
# =============================================================================
print("\nComputing metrics per stratum per seed...")

results = []
seeds = sorted(preds['seed'].unique())
excluded = []

for seed in seeds:
    seed_preds = preds[preds['seed'] == seed]
    
    for stratum in sorted(consistency['stratum'].unique()):
        stratum_preds = seed_preds[seed_preds['stratum'] == stratum]
        n = len(stratum_preds)
        
        if n == 0:
            continue
            
        y_true = stratum_preds['label'].values
        y_pred = stratum_preds['predicted'].values
        
        # Check if both classes present in ground truth
        if len(np.unique(y_true)) < 2:
            excluded.append({'stratum': stratum, 'seed': seed, 'n': n, 'reason': 'single_class'})
            continue
            
        results.append({
            'seed': seed,
            'stratum': stratum,
            'n': n,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_granted': f1_score(y_true, y_pred, pos_label=0),
            'f1_refused': f1_score(y_true, y_pred, pos_label=1),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro')
        })

results_df = pd.DataFrame(results)

# Report exclusions
if excluded:
    print(f"\nExcluded {len(excluded)} stratum-seed combinations (single class in ground truth):")
    exc_df = pd.DataFrame(excluded)
    for stratum in exc_df['stratum'].unique():
        seeds_exc = exc_df[exc_df['stratum'] == stratum]['seed'].tolist()
        print(f"  {stratum}: excluded in seeds {seeds_exc}")

# =============================================================================
# AGGREGATE ACROSS SEEDS
# =============================================================================
agg = results_df.groupby('stratum').agg({
    'n': ['first', 'count'],  # count = number of seeds included
    'accuracy': ['mean', 'std'],
    'f1_macro': ['mean', 'std'],
    'f1_granted': ['mean', 'std'],
    'f1_refused': ['mean', 'std'],
    'precision_macro': ['mean', 'std'],
    'recall_macro': ['mean', 'std']
}).reset_index()

agg.columns = ['_'.join(col).strip('_') for col in agg.columns]

# Sort by complexity (ALL THREE first, then 2-factor, then 1-factor, then UNKNOWN)
sort_order = {
    'ALL THREE': 0,
    'LAW + EVIDENCE': 1,
    'LAW + TRIAL': 2,
    'EVIDENCE + TRIAL': 3,
    'LAW only': 4,
    'EVIDENCE only': 5,
    'TRIAL only': 6,
    'UNKNOWN': 7
}
agg['sort_key'] = agg['stratum'].map(sort_order)
agg = agg.sort_values('sort_key').drop(columns=['sort_key'])

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "="*110)
print("TABLE: PERFORMANCE BY L/E/T STRATUM (averaged over 5 seeds, mean +/- std)")
print("="*110)

print(f"\n{'Stratum':<18} {'N':>5} {'Seeds':>5} {'Accuracy':>14} {'F1-Macro':>14} {'F1-Granted':>14} {'F1-Refused':>14}")
print("-"*110)

for _, row in agg.iterrows():
    stratum = row['stratum']
    n = int(row['n_first'])
    n_seeds = int(row['n_count'])
    acc = f"{row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f}"
    f1m = f"{row['f1_macro_mean']:.3f}+/-{row['f1_macro_std']:.3f}"
    f1g = f"{row['f1_granted_mean']:.3f}+/-{row['f1_granted_std']:.3f}"
    f1r = f"{row['f1_refused_mean']:.3f}+/-{row['f1_refused_std']:.3f}"
    
    print(f"{stratum:<18} {n:>5} {n_seeds:>5} {acc:>14} {f1m:>14} {f1g:>14} {f1r:>14}")

# Clean table (means only)
print("\n" + "="*110)
print("CLEAN TABLE (means only, for paper)")
print("="*110)

print(f"\n{'Stratum':<18} {'N':>6} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Granted':>12} {'F1-Refused':>12} {'Precision':>10} {'Recall':>10}")
print("-"*100)

for _, row in agg.iterrows():
    n_seeds = int(row['n_count'])
    note = "*" if n_seeds < 5 else ""
    print(f"{row['stratum']:<18} {int(row['n_first']):>5}{note} "
          f"{row['accuracy_mean']:>10.3f} {row['f1_macro_mean']:>10.3f} "
          f"{row['f1_granted_mean']:>12.3f} {row['f1_refused_mean']:>12.3f} "
          f"{row['precision_macro_mean']:>10.3f} {row['recall_macro_mean']:>10.3f}")

if any(agg['n_count'] < 5):
    print("\n* = fewer than 5 seeds included (some excluded due to single-class ground truth)")

# =============================================================================
# SAVE
# =============================================================================
agg.to_csv("multiseed_stratum_performance.csv", index=False)
print(f"\nSaved: multiseed_stratum_performance.csv")

# Paper-ready table
paper_table = []
for _, row in agg.iterrows():
    paper_table.append({
        'Stratum': row['stratum'],
        'N': int(row['n_first']),
        'Seeds_Included': int(row['n_count']),
        'Accuracy': round(row['accuracy_mean'], 3),
        'Accuracy_SD': round(row['accuracy_std'], 3),
        'F1_Macro': round(row['f1_macro_mean'], 3),
        'F1_Macro_SD': round(row['f1_macro_std'], 3),
        'F1_Granted': round(row['f1_granted_mean'], 3),
        'F1_Refused': round(row['f1_refused_mean'], 3),
        'Precision_Macro': round(row['precision_macro_mean'], 3),
        'Recall_Macro': round(row['recall_macro_mean'], 3),
    })

pd.DataFrame(paper_table).to_csv("multiseed_stratum_for_paper.csv", index=False)
print(f"Saved: multiseed_stratum_for_paper.csv")

print("\n" + "="*110)
print("DONE")
print("="*110)
