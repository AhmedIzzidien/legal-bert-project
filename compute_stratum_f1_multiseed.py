#!/usr/bin/env python3
"""
Comprehensive Multi-Seed CV Results Extraction (FIXED)

Fixes applied:
- Renamed 'label' to 'y_true' before merge to avoid ambiguity
- Aggregated confusion matrix clearly labeled as descriptive
- Per-fold iteration uses groupby for robustness
- FP/FN renamed to directional labels
- Added sanity checks and validation
- Added majority vote evaluation (one prediction per case)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import os

# =============================================================================
# LOAD AND VALIDATE DATA
# =============================================================================

print("="*80)
print("LOADING AND VALIDATING DATA")
print("="*80)

# Check files exist
assert os.path.exists('sj_231025.pkl'), "Missing: sj_231025.pkl"
assert os.path.exists('legalbert_multiseed_attention/all_predictions.csv'), "Missing: all_predictions.csv"
assert os.path.exists('legalbert_multiseed_attention/case_consistency.csv'), "Missing: case_consistency.csv"

# Load data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))
df['y_true'] = (df['outcome'] == 'summary judgment refused').astype(int)  # 0=granted, 1=refused

all_preds = pd.read_csv('legalbert_multiseed_attention/all_predictions.csv')
consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')

# Sanity checks
print(f"\nDataset: {len(df)} cases")
print(f"  Granted: {(df['y_true']==0).sum()} ({(df['y_true']==0).mean()*100:.1f}%)")
print(f"  Refused: {(df['y_true']==1).sum()} ({(df['y_true']==1).mean()*100:.1f}%)")

print(f"\nPredictions: {len(all_preds)} rows")
print(f"  Seeds: {sorted(all_preds['seed'].unique())}")
print(f"  Folds per seed: {all_preds.groupby('seed')['fold'].nunique().unique()}")

# Verify structure: each case appears once per seed
cases_per_seed = all_preds.groupby('seed')['case_id'].nunique()
print(f"  Cases per seed: {cases_per_seed.unique()}")
assert (cases_per_seed == len(df)).all(), "Mismatch: not all cases predicted per seed"

expected_rows = len(df) * len(all_preds['seed'].unique())
assert len(all_preds) == expected_rows, f"Expected {expected_rows} rows, got {len(all_preds)}"
print(f"  OK Validated: {len(df)} cases x {len(all_preds['seed'].unique())} seeds = {expected_rows} predictions")

# Merge with RENAMED columns to avoid ambiguity
# Check what columns are available for stratum analysis
stratum_cols_available = all(['law' in df.columns, 'evidence' in df.columns, 'trial' in df.columns])

if not stratum_cols_available:
    print("  INFO: 'law', 'evidence', 'trial' columns not found directly")
    print("  Extracting from 'decision_reason_categories_clean'...")
    
    # Format is: "LAW=1 EVIDENCE=1 TRIAL=0"
    if 'decision_reason_categories_clean' in df.columns:
        df['law'] = df['decision_reason_categories_clean'].str.extract(r'LAW=(\d)')[0].astype(int)
        df['evidence'] = df['decision_reason_categories_clean'].str.extract(r'EVIDENCE=(\d)')[0].astype(int)
        df['trial'] = df['decision_reason_categories_clean'].str.extract(r'TRIAL=(\d)')[0].astype(int)
        print(f"  OK Extracted L/E/T from decision_reason_categories_clean")
        print(f"     LAW=1: {df['law'].sum()} cases")
        print(f"     EVIDENCE=1: {df['evidence'].sum()} cases")
        print(f"     TRIAL=1: {df['trial'].sum()} cases")
    else:
        print("  WARNING: No decision_reason_categories_clean column found")
        print("  Creating dummy L/E/T columns (all zeros)")
        df['law'] = 0
        df['evidence'] = 0
        df['trial'] = 0

df_info = df[['case_id', 'y_true', 'law', 'evidence', 'trial']].copy()
n_before = len(all_preds)
all_preds = all_preds.merge(df_info, on='case_id', how='left')
assert len(all_preds) == n_before, f"Merge changed row count! {n_before} --> {len(all_preds)}"

# Check no missing merges
assert all_preds['y_true'].notna().all(), "Merge failed: missing y_true values"
print(f"  OK Merge successful: {len(all_preds)} rows preserved, no missing values")

# CRITICAL: Verify label consistency between all_predictions.csv and original data
# The CSV has 'label', we computed 'y_true' from df - they MUST match
if 'label' in all_preds.columns:
    label_match = (all_preds['label'] == all_preds['y_true']).all()
    if label_match:
        print(f"  OK Label consistency verified: CSV 'label' == computed 'y_true'")
    else:
        mismatch = (all_preds['label'] != all_preds['y_true']).sum()
        print(f"  WARNING WARNING: {mismatch} rows have label != y_true!")
        print(f"     This indicates data inconsistency - investigate before proceeding!")
        # Show examples
        bad = all_preds[all_preds['label'] != all_preds['y_true']].head()
        print(f"     Examples: {bad[['case_id', 'label', 'y_true']].to_dict('records')[:3]}")

# =============================================================================
# 1. OVERALL METRICS (per seed, then aggregated)
# =============================================================================

print("\n" + "="*80)
print("1. OVERALL PERFORMANCE METRICS")
print("="*80)

seed_metrics = []

for seed in sorted(all_preds['seed'].unique()):
    seed_preds = all_preds[all_preds['seed'] == seed]
    
    y_true = seed_preds['y_true']
    y_pred = seed_preds['predicted']
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_granted = f1_score(y_true, y_pred, pos_label=0)
    f1_refused = f1_score(y_true, y_pred, pos_label=1)
    prec_macro = precision_score(y_true, y_pred, average='macro')
    prec_granted = precision_score(y_true, y_pred, pos_label=0)
    prec_refused = precision_score(y_true, y_pred, pos_label=1)
    rec_macro = recall_score(y_true, y_pred, average='macro')
    rec_granted = recall_score(y_true, y_pred, pos_label=0)
    rec_refused = recall_score(y_true, y_pred, pos_label=1)
    
    seed_metrics.append({
        'seed': seed,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_granted': f1_granted,
        'f1_refused': f1_refused,
        'precision_macro': prec_macro,
        'precision_granted': prec_granted,
        'precision_refused': prec_refused,
        'recall_macro': rec_macro,
        'recall_granted': rec_granted,
        'recall_refused': rec_refused,
    })

seed_df = pd.DataFrame(seed_metrics)

print("\n1.1 Per-Seed Performance")
print("-"*100)
print(f"{'Seed':<8} {'Acc':<8} {'F1-Mac':<8} {'F1-Gr':<8} {'F1-Ref':<8} {'Prec-Gr':<9} {'Prec-Ref':<9} {'Rec-Gr':<8} {'Rec-Ref':<8}")
print("-"*100)
for _, row in seed_df.iterrows():
    print(f"{int(row['seed']):<8} {row['accuracy']:.3f}    {row['f1_macro']:.3f}    {row['f1_granted']:.3f}    {row['f1_refused']:.3f}    {row['precision_granted']:.3f}     {row['precision_refused']:.3f}     {row['recall_granted']:.3f}    {row['recall_refused']:.3f}")

print("\n1.2 Aggregated Performance (Mean +/- Std across 5 seeds)")
print("-"*70)
metrics_to_report = [
    ('Accuracy', 'accuracy'),
    ('F1-Macro', 'f1_macro'),
    ('F1-Granted', 'f1_granted'),
    ('F1-Refused', 'f1_refused'),
    ('Precision-Macro', 'precision_macro'),
    ('Precision-Granted', 'precision_granted'),
    ('Precision-Refused', 'precision_refused'),
    ('Recall-Macro', 'recall_macro'),
    ('Recall-Granted', 'recall_granted'),
    ('Recall-Refused', 'recall_refused'),
]

print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-"*60)
for name, col in metrics_to_report:
    mean = seed_df[col].mean()
    std = seed_df[col].std()
    min_val = seed_df[col].min()
    max_val = seed_df[col].max()
    print(f"{name:<20} {mean:.4f}     {std:.4f}     {min_val:.4f}     {max_val:.4f}")

# =============================================================================
# 2. PER-FOLD METRICS (using groupby for robustness)
# =============================================================================

print("\n" + "="*80)
print("2. PER-FOLD PERFORMANCE (all 25 folds)")
print("="*80)

fold_metrics = []

for (seed, fold), fold_preds in all_preds.groupby(['seed', 'fold']):
    y_true = fold_preds['y_true']
    y_pred = fold_preds['predicted']
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_granted = f1_score(y_true, y_pred, pos_label=0)
    f1_refused = f1_score(y_true, y_pred, pos_label=1)
    
    fold_metrics.append({
        'seed': seed,
        'fold': fold,
        'n_cases': len(fold_preds),
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_granted': f1_granted,
        'f1_refused': f1_refused,
    })

fold_df = pd.DataFrame(fold_metrics)

print(f"\n{'Seed':<8} {'Fold':<6} {'N':<6} {'Acc':<8} {'F1-Mac':<8} {'F1-Gr':<8} {'F1-Ref':<8}")
print("-"*60)
for _, row in fold_df.iterrows():
    print(f"{int(row['seed']):<8} {int(row['fold']):<6} {int(row['n_cases']):<6} {row['accuracy']:.3f}    {row['f1_macro']:.3f}    {row['f1_granted']:.3f}    {row['f1_refused']:.3f}")

print(f"\nFold Aggregates (n=25 folds):")
print(f"  Accuracy: {fold_df['accuracy'].mean():.4f} +/- {fold_df['accuracy'].std():.4f}")
print(f"  F1-Macro: {fold_df['f1_macro'].mean():.4f} +/- {fold_df['f1_macro'].std():.4f}")

# =============================================================================
# 3. CONFUSION MATRICES (per seed - the valid evaluation)
# =============================================================================

print("\n" + "="*80)
print("3. CONFUSION MATRICES (per seed)")
print("="*80)

print("\nNote: Each seed's confusion matrix represents independent evaluation.")
print("      Summing across seeds would double-count cases.\n")

seed_cms = []
for seed in sorted(all_preds['seed'].unique()):
    seed_preds = all_preds[all_preds['seed'] == seed]
    y_true = seed_preds['y_true']
    y_pred = seed_preds['predicted']
    cm = confusion_matrix(y_true, y_pred)
    seed_cms.append(cm)
    
    # Directional error rates
    rate_refused_as_granted = cm[1, 0] / cm[1, :].sum()  # True refused, pred granted
    rate_granted_as_refused = cm[0, 1] / cm[0, :].sum()  # True granted, pred refused
    
    print(f"Seed {seed}:")
    print(f"                 Pred Granted    Pred Refused")
    print(f"  True Granted   {cm[0,0]:>8}        {cm[0,1]:>8}")
    print(f"  True Refused   {cm[1,0]:>8}        {cm[1,1]:>8}")
    print(f"  Refused-->Granted: {rate_refused_as_granted:.1%}  |  Granted-->Refused: {rate_granted_as_refused:.1%}\n")

# Mean confusion matrix across seeds
mean_cm = np.mean(seed_cms, axis=0)
std_cm = np.std(seed_cms, axis=0)

print("Mean Confusion Matrix (+/- std across seeds):")
print(f"                 Pred Granted         Pred Refused")
print(f"  True Granted   {mean_cm[0,0]:.1f} +/- {std_cm[0,0]:.1f}      {mean_cm[0,1]:.1f} +/- {std_cm[0,1]:.1f}")
print(f"  True Refused   {mean_cm[1,0]:.1f} +/- {std_cm[1,0]:.1f}      {mean_cm[1,1]:.1f} +/- {std_cm[1,1]:.1f}")

# Mean error rates
mean_refused_as_granted = np.mean([cm[1,0]/cm[1,:].sum() for cm in seed_cms])
mean_granted_as_refused = np.mean([cm[0,1]/cm[0,:].sum() for cm in seed_cms])
std_refused_as_granted = np.std([cm[1,0]/cm[1,:].sum() for cm in seed_cms])
std_granted_as_refused = np.std([cm[0,1]/cm[0,:].sum() for cm in seed_cms])

print(f"\nMean Error Rates:")
print(f"  Refused-->Granted: {mean_refused_as_granted:.1%} +/- {std_refused_as_granted:.1%}")
print(f"  Granted-->Refused: {mean_granted_as_refused:.1%} +/- {std_granted_as_refused:.1%}")

# =============================================================================
# 4. MAJORITY VOTE EVALUATION (one prediction per case)
# =============================================================================

print("\n" + "="*80)
print("4. MAJORITY VOTE EVALUATION (ensemble across seeds)")
print("="*80)

# For each case, take majority vote across 5 seeds (deterministic: mean >= 0.5)
case_votes = all_preds.groupby('case_id').agg({
    'predicted': lambda x: int(x.mean() >= 0.5),  # Deterministic majority vote
    'y_true': 'first',  # Same for all rows of same case
}).reset_index()

y_true_vote = case_votes['y_true']
y_pred_vote = case_votes['predicted']

acc_vote = accuracy_score(y_true_vote, y_pred_vote)
f1_macro_vote = f1_score(y_true_vote, y_pred_vote, average='macro')
f1_granted_vote = f1_score(y_true_vote, y_pred_vote, pos_label=0)
f1_refused_vote = f1_score(y_true_vote, y_pred_vote, pos_label=1)
prec_granted_vote = precision_score(y_true_vote, y_pred_vote, pos_label=0)
prec_refused_vote = precision_score(y_true_vote, y_pred_vote, pos_label=1)
rec_granted_vote = recall_score(y_true_vote, y_pred_vote, pos_label=0)
rec_refused_vote = recall_score(y_true_vote, y_pred_vote, pos_label=1)
cm_vote = confusion_matrix(y_true_vote, y_pred_vote)

print(f"\nMajority Vote Performance (n={len(case_votes)} cases):")
print(f"  Accuracy:          {acc_vote:.4f}")
print(f"  F1-Macro:          {f1_macro_vote:.4f}")
print(f"  F1-Granted:        {f1_granted_vote:.4f}")
print(f"  F1-Refused:        {f1_refused_vote:.4f}")
print(f"  Precision-Granted: {prec_granted_vote:.4f}")
print(f"  Precision-Refused: {prec_refused_vote:.4f}")
print(f"  Recall-Granted:    {rec_granted_vote:.4f}")
print(f"  Recall-Refused:    {rec_refused_vote:.4f}")

print(f"\nMajority Vote Confusion Matrix:")
print(f"                 Pred Granted    Pred Refused")
print(f"  True Granted   {cm_vote[0,0]:>8}        {cm_vote[0,1]:>8}")
print(f"  True Refused   {cm_vote[1,0]:>8}        {cm_vote[1,1]:>8}")

rate_refused_as_granted = cm_vote[1, 0] / cm_vote[1, :].sum()
rate_granted_as_refused = cm_vote[0, 1] / cm_vote[0, :].sum()
print(f"\n  Refused-->Granted: {rate_refused_as_granted:.1%}")
print(f"  Granted-->Refused: {rate_granted_as_refused:.1%}")

# =============================================================================
# 5. CONSISTENCY ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("5. CONSISTENCY ANALYSIS")
print("="*80)

df_with_consistency = df.merge(consistency[['case_id', 'times_correct']], on='case_id', how='left')

# CRITICAL: Check no cases were lost
assert len(df_with_consistency) == len(df), f"Merge lost cases! {len(df)} --> {len(df_with_consistency)}"
missing_consistency = df_with_consistency['times_correct'].isna().sum()
if missing_consistency > 0:
    print(f"  WARNING FATAL: {missing_consistency} cases missing from case_consistency.csv!")
    missing_ids = df_with_consistency[df_with_consistency['times_correct'].isna()]['case_id'].tolist()
    print(f"     Missing case_ids: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
    raise ValueError("case_consistency.csv is incomplete - cannot proceed")
print(f"  OK All {len(df)} cases have consistency data")

print("\n5.1 Consistency Distribution")
print("-"*60)
print(f"{'Correct':<10} {'Cases':<10} {'Pct':<10} {'Interpretation':<30}")
print("-"*60)
for tc in range(6):
    n = len(df_with_consistency[df_with_consistency['times_correct'] == tc])
    pct = n / len(df_with_consistency) * 100
    if tc == 0:
        interp = "Always Wrong"
    elif tc == 5:
        interp = "Always Right"
    else:
        interp = f"Inconsistent ({tc}/5)"
    print(f"{tc}/5       {n:<10} {pct:.1f}%      {interp:<30}")

# By outcome
print("\n5.2 Consistency by Outcome")
print("-"*60)

for outcome, label_val in [('Granted', 0), ('Refused', 1)]:
    subset = df_with_consistency[df_with_consistency['y_true'] == label_val]
    print(f"\n{outcome} (n={len(subset)}):")
    print(f"  {'Correct':<10} {'Cases':<10} {'Pct':<10}")
    for tc in range(6):
        n = len(subset[subset['times_correct'] == tc])
        pct = n / len(subset) * 100
        print(f"  {tc}/5       {n:<10} {pct:.1f}%")
    
    always_wrong = len(subset[subset['times_correct'] == 0])
    always_right = len(subset[subset['times_correct'] == 5])
    print(f"  Always-Wrong Rate: {always_wrong/len(subset)*100:.1f}%")
    print(f"  Always-Right Rate: {always_right/len(subset)*100:.1f}%")

# =============================================================================
# 6. STRATUM ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("6. PERFORMANCE BY STRATUM (LAW/EVIDENCE/TRIAL)")
print("="*80)

# Create stratum labels
def get_stratum(row):
    parts = []
    if row['law'] == 1:
        parts.append('LAW')
    if row['evidence'] == 1:
        parts.append('EVIDENCE')
    if row['trial'] == 1:
        parts.append('TRIAL')
    if len(parts) == 0:
        return 'NONE'
    return ' + '.join(parts)

df_with_consistency['stratum'] = df_with_consistency.apply(get_stratum, axis=1)
all_preds['stratum'] = all_preds.apply(get_stratum, axis=1)

print("\n6.1 Per-Stratum Performance (mean across seeds)")
print("-"*100)

stratum_metrics = []
for stratum in sorted(df_with_consistency['stratum'].unique()):
    stratum_cases = df_with_consistency[df_with_consistency['stratum'] == stratum]
    
    # Compute per-seed metrics, then average
    seed_accs = []
    seed_f1s = []
    for seed in all_preds['seed'].unique():
        stratum_seed_preds = all_preds[(all_preds['stratum'] == stratum) & (all_preds['seed'] == seed)]
        if len(stratum_seed_preds) > 0:
            y_true = stratum_seed_preds['y_true']
            y_pred = stratum_seed_preds['predicted']
            seed_accs.append(accuracy_score(y_true, y_pred))
            seed_f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    always_wrong = len(stratum_cases[stratum_cases['times_correct'] == 0])
    always_right = len(stratum_cases[stratum_cases['times_correct'] == 5])
    
    stratum_metrics.append({
        'stratum': stratum,
        'n_cases': len(stratum_cases),
        'accuracy_mean': np.mean(seed_accs),
        'accuracy_std': np.std(seed_accs),
        'f1_macro_mean': np.mean(seed_f1s),
        'f1_macro_std': np.std(seed_f1s),
        'always_wrong': always_wrong,
        'always_wrong_pct': always_wrong / len(stratum_cases) * 100,
        'always_right': always_right,
        'always_right_pct': always_right / len(stratum_cases) * 100,
    })

stratum_df = pd.DataFrame(stratum_metrics).sort_values('accuracy_mean', ascending=False)

print(f"{'Stratum':<25} {'N':<6} {'Acc (mean+/-std)':<18} {'F1 (mean+/-std)':<18} {'AW%':<8} {'AR%':<8}")
print("-"*100)
for _, row in stratum_df.iterrows():
    print(f"{row['stratum']:<25} {int(row['n_cases']):<6} {row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f}        {row['f1_macro_mean']:.3f}+/-{row['f1_macro_std']:.3f}        {row['always_wrong_pct']:.1f}%    {row['always_right_pct']:.1f}%")

print("\n6.2 Stratum x Outcome Analysis")
print("-"*80)

for stratum in sorted(df_with_consistency['stratum'].unique()):
    stratum_cases = df_with_consistency[df_with_consistency['stratum'] == stratum]
    
    print(f"\n{stratum} (n={len(stratum_cases)}):")
    
    for outcome, label_val in [('Granted', 0), ('Refused', 1)]:
        subset = stratum_cases[stratum_cases['y_true'] == label_val]
        if len(subset) == 0:
            continue
        
        always_wrong = len(subset[subset['times_correct'] == 0])
        always_right = len(subset[subset['times_correct'] == 5])
        
        print(f"  {outcome}: n={len(subset)}, AW={always_wrong} ({always_wrong/len(subset)*100:.1f}%), AR={always_right} ({always_right/len(subset)*100:.1f}%)")

# =============================================================================
# 7. NUMBER OF GROUNDS ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("7. PERFORMANCE BY NUMBER OF GROUNDS")
print("="*80)

df_with_consistency['n_grounds'] = df_with_consistency['law'] + df_with_consistency['evidence'] + df_with_consistency['trial']
all_preds_merged = all_preds.merge(df_with_consistency[['case_id', 'n_grounds', 'times_correct']], on='case_id', how='left')
assert len(all_preds_merged) == len(all_preds), f"Merge lost predictions! {len(all_preds)} --> {len(all_preds_merged)}"
assert all_preds_merged['n_grounds'].notna().all(), "Missing n_grounds after merge!"

print("\n7.1 Accuracy by Number of Grounds")
print("-"*80)
print(f"{'N Grounds':<12} {'N Cases':<10} {'Acc (mean+/-std)':<18} {'AW%':<10} {'AR%':<10}")
print("-"*80)

for n_grounds in sorted(df_with_consistency['n_grounds'].unique()):
    subset_cases = df_with_consistency[df_with_consistency['n_grounds'] == n_grounds]
    
    # Per-seed accuracy
    seed_accs = []
    for seed in all_preds['seed'].unique():
        subset_preds = all_preds_merged[(all_preds_merged['n_grounds'] == n_grounds) & (all_preds_merged['seed'] == seed)]
        if len(subset_preds) > 0:
            seed_accs.append(accuracy_score(subset_preds['y_true'], subset_preds['predicted']))
    
    aw = len(subset_cases[subset_cases['times_correct'] == 0])
    ar = len(subset_cases[subset_cases['times_correct'] == 5])
    
    print(f"{n_grounds:<12} {len(subset_cases):<10} {np.mean(seed_accs):.3f}+/-{np.std(seed_accs):.3f}        {aw/len(subset_cases)*100:.1f}%      {ar/len(subset_cases)*100:.1f}%")

# =============================================================================
# 8. DETAILED ERROR ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("8. DETAILED ERROR ANALYSIS")
print("="*80)

always_wrong_cases = df_with_consistency[df_with_consistency['times_correct'] == 0]
always_right_cases = df_with_consistency[df_with_consistency['times_correct'] == 5]

print(f"\n8.1 Always-Wrong Cases (n={len(always_wrong_cases)})")
print("-"*60)

# By outcome
aw_granted = len(always_wrong_cases[always_wrong_cases['y_true'] == 0])
aw_refused = len(always_wrong_cases[always_wrong_cases['y_true'] == 1])
print(f"  By True Outcome:")
print(f"    Granted (model predicted Refused): {aw_granted} ({aw_granted/len(always_wrong_cases)*100:.1f}%)")
print(f"    Refused (model predicted Granted): {aw_refused} ({aw_refused/len(always_wrong_cases)*100:.1f}%)")

# By stratum
print(f"\n  By Stratum:")
for stratum in sorted(always_wrong_cases['stratum'].unique()):
    n = len(always_wrong_cases[always_wrong_cases['stratum'] == stratum])
    total_stratum = len(df_with_consistency[df_with_consistency['stratum'] == stratum])
    print(f"    {stratum}: {n} ({n/total_stratum*100:.1f}% of stratum)")

print(f"\n8.2 Always-Right Cases (n={len(always_right_cases)})")
print("-"*60)

# By outcome
ar_granted = len(always_right_cases[always_right_cases['y_true'] == 0])
ar_refused = len(always_right_cases[always_right_cases['y_true'] == 1])
print(f"  By True Outcome:")
print(f"    Granted: {ar_granted} ({ar_granted/len(always_right_cases)*100:.1f}%)")
print(f"    Refused: {ar_refused} ({ar_refused/len(always_right_cases)*100:.1f}%)")

# =============================================================================
# 9. COMPARISON WITH SINGLE-SEED BASELINE
# =============================================================================

print("\n" + "="*80)
print("9. COMPARISON: SINGLE-SEED vs MULTI-SEED")
print("="*80)

print("\n" + "-"*80)
print(f"{'Metric':<25} {'Single-Seed':<15} {'Multi-Seed (Mean+/-Std)':<25} {'Diff':<10}")
print("-"*80)

# Single-seed values from initial report
single_seed = {
    'Accuracy': 0.668,
    'F1-Macro': 0.657,
    'F1-Granted': 0.719,
    'F1-Refused': 0.594,
}

multi_seed = {
    'Accuracy': (seed_df['accuracy'].mean(), seed_df['accuracy'].std()),
    'F1-Macro': (seed_df['f1_macro'].mean(), seed_df['f1_macro'].std()),
    'F1-Granted': (seed_df['f1_granted'].mean(), seed_df['f1_granted'].std()),
    'F1-Refused': (seed_df['f1_refused'].mean(), seed_df['f1_refused'].std()),
}

for metric in single_seed:
    ss = single_seed[metric]
    ms_mean, ms_std = multi_seed[metric]
    diff = ms_mean - ss
    print(f"{metric:<25} {ss:.3f}           {ms_mean:.3f} +/- {ms_std:.3f}            {diff:+.3f}")

print("\nInterpretation:")
print("  - Multi-seed CV provides more realistic performance estimates")
print("  - Single-seed may have benefited from favorable train/test split")
print("  - Differences indicate initial report overestimated performance")

# =============================================================================
# 10. SUMMARY TABLES FOR PAPER
# =============================================================================

print("\n" + "="*80)
print("10. SUMMARY TABLES FOR PAPER")
print("="*80)

print("\n" + "="*60)
print("TABLE 1: Overall Performance (5 seeds x 5 folds)")
print("="*60)
print(f"{'Metric':<25} {'Value':<20}")
print("-"*45)
print(f"{'Accuracy':<25} {seed_df['accuracy'].mean()*100:.1f}% +/- {seed_df['accuracy'].std()*100:.1f}%")
print(f"{'F1-Macro':<25} {seed_df['f1_macro'].mean():.3f} +/- {seed_df['f1_macro'].std():.3f}")
print(f"{'F1-Granted':<25} {seed_df['f1_granted'].mean():.3f} +/- {seed_df['f1_granted'].std():.3f}")
print(f"{'F1-Refused':<25} {seed_df['f1_refused'].mean():.3f} +/- {seed_df['f1_refused'].std():.3f}")
print(f"{'Precision-Granted':<25} {seed_df['precision_granted'].mean():.3f} +/- {seed_df['precision_granted'].std():.3f}")
print(f"{'Precision-Refused':<25} {seed_df['precision_refused'].mean():.3f} +/- {seed_df['precision_refused'].std():.3f}")
print(f"{'Recall-Granted':<25} {seed_df['recall_granted'].mean():.3f} +/- {seed_df['recall_granted'].std():.3f}")
print(f"{'Recall-Refused':<25} {seed_df['recall_refused'].mean():.3f} +/- {seed_df['recall_refused'].std():.3f}")

print("\n" + "="*60)
print("TABLE 2: Consistency Distribution")
print("="*60)
total = len(df_with_consistency)
for tc in range(6):
    n = len(df_with_consistency[df_with_consistency['times_correct'] == tc])
    print(f"{tc}/5 correct: {n} ({n/total*100:.1f}%)")

print("\n" + "="*60)
print("TABLE 3: Error Rate by Outcome")
print("="*60)
for outcome, label_val in [('Granted', 0), ('Refused', 1)]:
    subset = df_with_consistency[df_with_consistency['y_true'] == label_val]
    aw = len(subset[subset['times_correct'] == 0])
    print(f"{outcome}: {aw}/{len(subset)} always-wrong ({aw/len(subset)*100:.1f}%)")

print("\n" + "="*60)
print("TABLE 4: Performance by Number of Grounds")
print("="*60)
for n_grounds in sorted(df_with_consistency['n_grounds'].unique()):
    subset_cases = df_with_consistency[df_with_consistency['n_grounds'] == n_grounds]
    seed_accs = []
    for seed in all_preds['seed'].unique():
        subset_preds = all_preds_merged[(all_preds_merged['n_grounds'] == n_grounds) & (all_preds_merged['seed'] == seed)]
        if len(subset_preds) > 0:
            seed_accs.append(accuracy_score(subset_preds['y_true'], subset_preds['predicted']))
    aw = len(subset_cases[subset_cases['times_correct'] == 0])
    ar = len(subset_cases[subset_cases['times_correct'] == 5])
    print(f"{n_grounds} grounds (n={len(subset_cases)}): Acc={np.mean(seed_accs)*100:.1f}%+/-{np.std(seed_accs)*100:.1f}%, AW={aw/len(subset_cases)*100:.1f}%, AR={ar/len(subset_cases)*100:.1f}%")

print("\n" + "="*60)
print("TABLE 5: Majority Vote Performance")
print("="*60)
print(f"Accuracy:          {acc_vote*100:.1f}%")
print(f"F1-Macro:          {f1_macro_vote:.3f}")
print(f"F1-Granted:        {f1_granted_vote:.3f}")
print(f"F1-Refused:        {f1_refused_vote:.3f}")
print(f"Precision-Granted: {prec_granted_vote:.3f}")
print(f"Precision-Refused: {prec_refused_vote:.3f}")
print(f"Recall-Granted:    {rec_granted_vote:.3f}")
print(f"Recall-Refused:    {rec_refused_vote:.3f}")

# =============================================================================
# 11. SAVE DETAILED RESULTS TO CSV
# =============================================================================

print("\n" + "="*80)
print("11. LINGUISTIC MARKERS ANALYSIS")
print("="*80)

# Combine all text fields
df_with_consistency['all_text'] = (
    df_with_consistency['facts'].fillna('') + ' ' + 
    df_with_consistency['applicant_reason'].fillna('') + ' ' + 
    df_with_consistency['defence_reason'].fillna('')
).str.lower()

# Define phrases to analyze
phrases = [
    'factual dispute', 'oral evidence', 'no real prospect', 
    'compelling reason', 'arguable', 'bound to fail', 
    'no defence', 'summary judgment', 'triable issue',
    'real prospect', 'fanciful', 'credibility'
]

print("\nTable 4: Phrase frequency in errors vs. correct classifications")
print("-"*80)
print(f"{'Phrase':<25} {'In Errors':<12} {'In Correct':<12} {'Ratio':<10} {'Interpretation'}")
print("-"*80)

phrase_results = []
for phrase in phrases:
    # Check if phrase appears in text
    df_with_consistency[f'has_{phrase.replace(" ", "_")}'] = df_with_consistency['all_text'].str.contains(phrase, regex=False).astype(int)
    
    # Calculate frequencies
    errors = df_with_consistency[df_with_consistency['times_correct'] == 0]
    correct = df_with_consistency[df_with_consistency['times_correct'] == 5]
    
    freq_errors = errors[f'has_{phrase.replace(" ", "_")}'].mean() * 100 if len(errors) > 0 else 0
    freq_correct = correct[f'has_{phrase.replace(" ", "_")}'].mean() * 100 if len(correct) > 0 else 0
    
    if freq_correct > 0:
        ratio = freq_errors / freq_correct
    else:
        ratio = float('inf') if freq_errors > 0 else 1.0
    
    interp = "More in errors" if ratio > 1.2 else "More in correct" if ratio < 0.8 else "Similar"
    
    phrase_results.append({
        'phrase': phrase,
        'freq_errors': freq_errors,
        'freq_correct': freq_correct,
        'ratio': ratio
    })
    
    if ratio != float('inf'):
        print(f"{phrase:<25} {freq_errors:>6.1f}%      {freq_correct:>6.1f}%      {ratio:>5.2f}x    {interp}")
    else:
        print(f"{phrase:<25} {freq_errors:>6.1f}%      {freq_correct:>6.1f}%      {'inf':>5}     {interp}")

# =============================================================================
# 12. PERFORMANCE BY AREA OF LAW (TOPIC)
# =============================================================================

print("\n" + "="*80)
print("12. PERFORMANCE BY AREA OF LAW")
print("="*80)

# Check if topic columns exist
if 'primary_topic' in df.columns:
    df_with_consistency = df_with_consistency.merge(
        df[['case_id', 'primary_topic']], on='case_id', how='left'
    )
    
    print("\nTable 7: Performance by area of law")
    print("-"*80)
    print(f"{'Area of Law':<25} {'Cases':<8} {'AW':<6} {'AR':<6} {'Error Rate':<12} {'AR Rate':<10}")
    print("-"*80)
    
    topic_results = []
    for topic in df_with_consistency['primary_topic'].dropna().unique():
        subset = df_with_consistency[df_with_consistency['primary_topic'] == topic]
        if len(subset) < 10:  # Skip very small categories
            continue
        
        aw = len(subset[subset['times_correct'] == 0])
        ar = len(subset[subset['times_correct'] == 5])
        
        topic_results.append({
            'topic': topic,
            'n_cases': len(subset),
            'always_wrong': aw,
            'always_right': ar,
            'error_rate': aw / len(subset) * 100,
            'ar_rate': ar / len(subset) * 100
        })
    
    # Sort by error rate descending
    topic_results = sorted(topic_results, key=lambda x: x['error_rate'], reverse=True)
    
    for r in topic_results:
        print(f"{r['topic']:<25} {r['n_cases']:<8} {r['always_wrong']:<6} {r['always_right']:<6} {r['error_rate']:>6.1f}%      {r['ar_rate']:>6.1f}%")
else:
    print("\n  WARNING 'primary_topic' column not found in dataset")
    print("  Checking for alternative topic columns...")
    topic_cols = [c for c in df.columns if 'topic' in c.lower()]
    print(f"  Available: {topic_cols if topic_cols else 'None'}")

# =============================================================================
# 13. FALSE POSITIVE / FALSE NEGATIVE BREAKDOWN BY STRATUM
# =============================================================================

print("\n" + "="*80)
print("13. FP/FN BREAKDOWN BY STRATUM (Appendix C)")
print("="*80)

# For always-wrong cases, determine if FP or FN
# FP = predicted Granted (0), actually Refused (1) --> label=1
# FN = predicted Refused (1), actually Granted (0) --> label=0

always_wrong = df_with_consistency[df_with_consistency['times_correct'] == 0].copy()

# FN: True=Granted (label=0), model consistently predicted Refused
fn_cases = always_wrong[always_wrong['y_true'] == 0]
# FP: True=Refused (label=1), model consistently predicted Granted  
fp_cases = always_wrong[always_wrong['y_true'] == 1]

print(f"\nTotal Always-Wrong: {len(always_wrong)}")
print(f"  False Negatives (True=Granted, Pred=Refused): {len(fn_cases)}")
print(f"  False Positives (True=Refused, Pred=Granted): {len(fp_cases)}")

print("\n" + "-"*60)
print("FALSE NEGATIVES by Stratum (predicted Refused, actually Granted)")
print("-"*60)
print(f"{'Stratum':<30} {'Count':<10} {'% of FN':<10}")
print("-"*50)

for stratum in sorted(fn_cases['stratum'].unique()):
    n = len(fn_cases[fn_cases['stratum'] == stratum])
    pct = n / len(fn_cases) * 100 if len(fn_cases) > 0 else 0
    print(f"{stratum:<30} {n:<10} {pct:>6.1f}%")

print("\n" + "-"*60)
print("FALSE POSITIVES by Stratum (predicted Granted, actually Refused)")
print("-"*60)
print(f"{'Stratum':<30} {'Count':<10} {'% of FP':<10}")
print("-"*50)

for stratum in sorted(fp_cases['stratum'].unique()):
    n = len(fp_cases[fp_cases['stratum'] == stratum])
    pct = n / len(fp_cases) * 100 if len(fp_cases) > 0 else 0
    print(f"{stratum:<30} {n:<10} {pct:>6.1f}%")

# =============================================================================
# 14. DISTRIBUTION TABLE (Table A1)
# =============================================================================

print("\n" + "="*80)
print("14. DECISION FACTOR DISTRIBUTION (Table A1)")
print("="*80)

print("\n" + "-"*80)
print(f"{'LAW':<6} {'EVIDENCE':<10} {'TRIAL':<8} {'Count':<10} {'%':<8} {'Interpretation'}")
print("-"*80)

# Create all 8 combinations
for l in [0, 1]:
    for e in [0, 1]:
        for t in [0, 1]:
            subset = df_with_consistency[
                (df_with_consistency['law'] == l) & 
                (df_with_consistency['evidence'] == e) & 
                (df_with_consistency['trial'] == t)
            ]
            n = len(subset)
            pct = n / len(df_with_consistency) * 100
            
            # Interpretation
            parts = []
            if l: parts.append('Law')
            if e: parts.append('Evidence')
            if t: parts.append('Trial')
            interp = ' + '.join(parts) if parts else 'None cited'
            
            print(f"{l:<6} {e:<10} {t:<8} {n:<10} {pct:>5.1f}%   {interp}")

# =============================================================================
# 15. SAVE ALL RESULTS TO CSV
# =============================================================================

print("\n" + "="*80)
print("15. SAVING DETAILED RESULTS")
print("="*80)

# Save all metrics
seed_df.to_csv('multiseed_per_seed_metrics.csv', index=False)
fold_df.to_csv('multiseed_per_fold_metrics.csv', index=False)
stratum_df.to_csv('multiseed_per_stratum_metrics.csv', index=False)

print("\nSaved:")
print("  - multiseed_per_seed_metrics.csv")
print("  - multiseed_per_fold_metrics.csv")
print("  - multiseed_per_stratum_metrics.csv")

# =============================================================================
# 16. DOMAIN-SPECIFIC KEYWORD ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("16. DOMAIN-SPECIFIC KEYWORD ANALYSIS (Domain-Conditioned Bias)")
print("="*80)

# Keywords identified as associated with errors in statistical analysis
domain_keywords = {
    'binding': r'\bbinding\b',
    'defamatory': r'\bdefamatory\b',
    'settlement': r'\bsettlement\b',
    'guarantee': r'\bguarantee\b',
    'property': r'\bproperty\b',
    'jurisdiction': r'\bjurisdiction\b',
}

print("\nKeyword x Outcome x Error Rate Analysis:")
print("-"*90)
print(f"{'Keyword':<15} {'Has KW':<8} {'AW':<6} {'AW%':<8} {'Refused+KW':<12} {'R+KW AW':<10} {'R+KW AW%':<10}")
print("-"*90)

for keyword, pattern in domain_keywords.items():
    df_with_consistency[f'has_{keyword}'] = df_with_consistency['all_text'].str.contains(pattern, case=False, regex=True).astype(int)
    
    # Overall for keyword
    has_kw = df_with_consistency[df_with_consistency[f'has_{keyword}'] == 1]
    has_kw_aw = len(has_kw[has_kw['times_correct'] == 0])
    has_kw_pct = has_kw_aw / len(has_kw) * 100 if len(has_kw) > 0 else 0
    
    # Refused + keyword (domain-conditioned)
    refused_kw = df_with_consistency[(df_with_consistency[f'has_{keyword}'] == 1) & (df_with_consistency['y_true'] == 1)]
    refused_kw_aw = len(refused_kw[refused_kw['times_correct'] == 0])
    refused_kw_pct = refused_kw_aw / len(refused_kw) * 100 if len(refused_kw) > 0 else 0
    
    print(f"{keyword:<15} {len(has_kw):<8} {has_kw_aw:<6} {has_kw_pct:.1f}%    {len(refused_kw):<12} {refused_kw_aw:<10} {refused_kw_pct:.1f}%")

print("\nNote: 'binding' + REFUSED shows elevated error rate (domain-conditioned bias)")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)

