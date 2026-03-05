#!/usr/bin/env python3
"""
===============================================================================
ANALYZE ALWAYS-WRONG CASES (0/5 correct)
===============================================================================

Compares the 153 cases that ALWAYS fail vs cases that ALWAYS succeed.
Looks for patterns in text, attention, and case characteristics.

Usage:
    python analyze_failures.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

# ===============================================================================
# CONFIGURATION
# ===============================================================================

DATA_DIR = Path("legalbert_multiseed_attention")
ORIGINAL_DATA = "sj_231025.pkl"

# ===============================================================================
# LOAD DATA
# ===============================================================================

print("\n" + "="*70)
print("📂 LOADING DATA")
print("="*70)

# Load consistency data
case_consistency = pd.read_csv(DATA_DIR / "case_consistency.csv")
attention_per_case = pd.read_csv(DATA_DIR / "attention_per_case.csv")

# Load original data for text
df_original = pd.read_pickle(ORIGINAL_DATA)
outcomes = ["summary judgment granted", "summary judgment refused"]
df_original = df_original[df_original["outcome"].isin(outcomes)].copy()
df_original["label"] = df_original["outcome"].map({
    "summary judgment granted": 0, 
    "summary judgment refused": 1
})
df_original["case_id"] = range(len(df_original))

print(f"   Total cases: {len(case_consistency)}")

# ===============================================================================
# SPLIT BY CONSISTENCY
# ===============================================================================

always_wrong = case_consistency[case_consistency['times_correct'] == 0].copy()
always_right = case_consistency[case_consistency['times_correct'] == 5].copy()
sometimes_wrong = case_consistency[(case_consistency['times_correct'] > 0) & 
                                    (case_consistency['times_correct'] < 5)].copy()

print(f"\n   Always wrong (0/5): {len(always_wrong)}")
print(f"   Always right (5/5): {len(always_right)}")
print(f"   Sometimes wrong:    {len(sometimes_wrong)}")

# ===============================================================================
# ANALYSIS 1: BREAKDOWN BY LABEL
# ===============================================================================

print("\n" + "="*70)
print("📊 ANALYSIS 1: BREAKDOWN BY LABEL")
print("="*70)

print("\n   ALWAYS WRONG (0/5):")
label_dist = always_wrong['label'].value_counts()
for label, count in label_dist.items():
    label_name = "Granted" if label == 0 else "Refused"
    pct = count / len(always_wrong) * 100
    print(f"      {label_name}: {count} ({pct:.1f}%)")

print("\n   ALWAYS RIGHT (5/5):")
label_dist = always_right['label'].value_counts()
for label, count in label_dist.items():
    label_name = "Granted" if label == 0 else "Refused"
    pct = count / len(always_right) * 100
    print(f"      {label_name}: {count} ({pct:.1f}%)")

# What's the error type?
print("\n   ERROR TYPE (Always Wrong):")
# Merge with attention to get predicted
always_wrong_attn = always_wrong.merge(
    attention_per_case[['case_id', 'times_correct']].drop_duplicates(),
    on='case_id',
    how='left',
    suffixes=('', '_attn')
)

# Get predictions from all_predictions
all_preds = pd.read_csv(DATA_DIR / "all_predictions.csv")
wrong_preds = all_preds[all_preds['case_id'].isin(always_wrong['case_id'])]

# Majority prediction for each case
majority_pred = wrong_preds.groupby('case_id').agg({
    'predicted': lambda x: x.mode().iloc[0],
    'label': 'first'
}).reset_index()

fp = len(majority_pred[(majority_pred['label'] == 1) & (majority_pred['predicted'] == 0)])
fn = len(majority_pred[(majority_pred['label'] == 0) & (majority_pred['predicted'] == 1)])

print(f"      False Positives (pred Granted, was Refused): {fp}")
print(f"      False Negatives (pred Refused, was Granted): {fn}")

# ===============================================================================
# ANALYSIS 2: BREAKDOWN BY STRATUM
# ===============================================================================

print("\n" + "="*70)
print("📊 ANALYSIS 2: BREAKDOWN BY STRATUM")
print("="*70)

print("\n   ALWAYS WRONG by stratum:")
stratum_wrong = always_wrong['stratum'].value_counts()
for stratum, count in stratum_wrong.items():
    pct = count / len(always_wrong) * 100
    print(f"      {stratum:<20}: {count:>4} ({pct:>5.1f}%)")

print("\n   ALWAYS RIGHT by stratum:")
stratum_right = always_right['stratum'].value_counts()
for stratum, count in stratum_right.items():
    pct = count / len(always_right) * 100
    print(f"      {stratum:<20}: {count:>4} ({pct:>5.1f}%)")

# Ratio comparison
print("\n   FAILURE RATE by stratum (always wrong / total in stratum):")
all_strata = case_consistency.groupby('stratum').size()
wrong_by_stratum = always_wrong.groupby('stratum').size()

for stratum in all_strata.index:
    total = all_strata[stratum]
    wrong = wrong_by_stratum.get(stratum, 0)
    rate = wrong / total * 100
    print(f"      {stratum:<20}: {wrong:>3}/{total:<4} = {rate:>5.1f}% failure")

# ===============================================================================
# ANALYSIS 3: TEXT LENGTH
# ===============================================================================

print("\n" + "="*70)
print("📊 ANALYSIS 3: TEXT LENGTH")
print("="*70)

# Merge with original data
always_wrong_full = always_wrong.merge(df_original[['case_id', 'facts', 'applicant_reason', 'defence_reason']], 
                                        on='case_id', how='left')
always_right_full = always_right.merge(df_original[['case_id', 'facts', 'applicant_reason', 'defence_reason']], 
                                        on='case_id', how='left')

def get_text_stats(df, name):
    facts_len = df['facts'].fillna('').str.len()
    app_len = df['applicant_reason'].fillna('').str.len()
    def_len = df['defence_reason'].fillna('').str.len()
    total_len = facts_len + app_len + def_len
    
    print(f"\n   {name}:")
    print(f"      Total text length:    mean={total_len.mean():.0f}, median={total_len.median():.0f}")
    print(f"      Facts length:         mean={facts_len.mean():.0f}, median={facts_len.median():.0f}")
    print(f"      Applicant length:     mean={app_len.mean():.0f}, median={app_len.median():.0f}")
    print(f"      Defence length:       mean={def_len.mean():.0f}, median={def_len.median():.0f}")
    
    # Missing fields
    facts_missing = (df['facts'].isna() | (df['facts'].str.len() < 10)).sum()
    app_missing = (df['applicant_reason'].isna() | (df['applicant_reason'].str.len() < 10)).sum()
    def_missing = (df['defence_reason'].isna() | (df['defence_reason'].str.len() < 10)).sum()
    
    print(f"      Missing/empty facts:      {facts_missing} ({facts_missing/len(df)*100:.1f}%)")
    print(f"      Missing/empty applicant:  {app_missing} ({app_missing/len(df)*100:.1f}%)")
    print(f"      Missing/empty defence:    {def_missing} ({def_missing/len(df)*100:.1f}%)")

get_text_stats(always_wrong_full, "ALWAYS WRONG")
get_text_stats(always_right_full, "ALWAYS RIGHT")

# ===============================================================================
# ANALYSIS 4: ATTENTION COMPARISON
# ===============================================================================

print("\n" + "="*70)
print("📊 ANALYSIS 4: ATTENTION PATTERNS")
print("="*70)

attn_wrong = attention_per_case[attention_per_case['case_id'].isin(always_wrong['case_id'])]
attn_right = attention_per_case[attention_per_case['case_id'].isin(always_right['case_id'])]

print("\n   ALWAYS WRONG attention:")
print(f"      FACTS:     {attn_wrong['mean_attn_FACTS'].mean():.1%}")
print(f"      APPLICANT: {attn_wrong['mean_attn_APPLICANT'].mean():.1%}")
print(f"      DEFENCE:   {attn_wrong['mean_attn_DEFENCE'].mean():.1%}")

print("\n   ALWAYS RIGHT attention:")
print(f"      FACTS:     {attn_right['mean_attn_FACTS'].mean():.1%}")
print(f"      APPLICANT: {attn_right['mean_attn_APPLICANT'].mean():.1%}")
print(f"      DEFENCE:   {attn_right['mean_attn_DEFENCE'].mean():.1%}")

# By label within always wrong
print("\n   ALWAYS WRONG by label:")
wrong_granted = attn_wrong[attn_wrong['label'] == 0]
wrong_refused = attn_wrong[attn_wrong['label'] == 1]

if len(wrong_granted) > 0:
    print(f"\n      Granted (n={len(wrong_granted)}):")
    print(f"         FACTS:     {wrong_granted['mean_attn_FACTS'].mean():.1%}")
    print(f"         APPLICANT: {wrong_granted['mean_attn_APPLICANT'].mean():.1%}")
    print(f"         DEFENCE:   {wrong_granted['mean_attn_DEFENCE'].mean():.1%}")

if len(wrong_refused) > 0:
    print(f"\n      Refused (n={len(wrong_refused)}):")
    print(f"         FACTS:     {wrong_refused['mean_attn_FACTS'].mean():.1%}")
    print(f"         APPLICANT: {wrong_refused['mean_attn_APPLICANT'].mean():.1%}")
    print(f"         DEFENCE:   {wrong_refused['mean_attn_DEFENCE'].mean():.1%}")

# ===============================================================================
# ANALYSIS 5: COMMON WORDS IN FAILURES
# ===============================================================================

print("\n" + "="*70)
print("📊 ANALYSIS 5: TEXT PATTERNS IN FAILURES")
print("="*70)

def get_word_freq(texts):
    all_words = []
    for text in texts:
        if pd.notna(text):
            words = re.findall(r'\b[a-z]+\b', str(text).lower())
            all_words.extend(words)
    return Counter(all_words)

# Stop words to ignore
stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
              'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
              'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
              'into', 'through', 'during', 'before', 'after', 'above', 'below',
              'between', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
              'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
              'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
              'that', 'these', 'those', 'am', 'it', 'its', 'he', 'she', 'they',
              'them', 'his', 'her', 'their', 'what', 'which', 'who', 'whom'}

# Get combined text
wrong_texts = (always_wrong_full['facts'].fillna('') + ' ' + 
               always_wrong_full['applicant_reason'].fillna('') + ' ' +
               always_wrong_full['defence_reason'].fillna(''))

right_texts = (always_right_full['facts'].fillna('') + ' ' + 
               always_right_full['applicant_reason'].fillna('') + ' ' +
               always_right_full['defence_reason'].fillna(''))

wrong_words = get_word_freq(wrong_texts)
right_words = get_word_freq(right_texts)

# Normalize by document count
wrong_freq = {w: c/len(always_wrong) for w, c in wrong_words.items() if w not in stop_words}
right_freq = {w: c/len(always_right) for w, c in right_words.items() if w not in stop_words}

# Find words more common in failures
print("\n   Words MORE COMMON in ALWAYS WRONG (vs ALWAYS RIGHT):")
overrepresented = []
for word, freq in wrong_freq.items():
    if freq > 0.1:  # Appears in >10% of wrong cases
        right_f = right_freq.get(word, 0.001)
        ratio = freq / right_f
        if ratio > 1.5:  # 50% more common in failures
            overrepresented.append((word, freq, right_f, ratio))

overrepresented.sort(key=lambda x: x[3], reverse=True)
for word, wrong_f, right_f, ratio in overrepresented[:15]:
    print(f"      '{word}': {wrong_f:.1%} in wrong vs {right_f:.1%} in right ({ratio:.1f}x)")

# ===============================================================================
# ANALYSIS 6: SPECIFIC CASE EXAMPLES
# ===============================================================================

print("\n" + "="*70)
print("📊 ANALYSIS 6: SAMPLE FAILED CASES")
print("="*70)

# Get 5 random always-wrong cases
sample_wrong = always_wrong_full.sample(min(5, len(always_wrong_full)), random_state=42)

for i, (_, row) in enumerate(sample_wrong.iterrows()):
    print(f"\n   --- Case {i+1} (ID: {row['case_id']}) ---")
    print(f"   Label: {'Granted' if row['label'] == 0 else 'Refused'}")
    print(f"   Stratum: {row['stratum']}")
    
    # Get prediction info
    case_preds = all_preds[all_preds['case_id'] == row['case_id']]
    pred_mode = case_preds['predicted'].mode().iloc[0]
    print(f"   Model predicted: {'Granted' if pred_mode == 0 else 'Refused'} (all 5 seeds)")
    
    facts_preview = str(row['facts'])[:200] + "..." if pd.notna(row['facts']) else "[EMPTY]"
    print(f"   Facts preview: {facts_preview}")

# ===============================================================================
# SAVE DETAILED FAILURE ANALYSIS
# ===============================================================================

print("\n" + "="*70)
print("💾 SAVING DETAILED ANALYSIS")
print("="*70)

# Save always-wrong cases with full info
always_wrong_export = always_wrong.merge(
    df_original[['case_id', 'facts', 'applicant_reason', 'defence_reason', 'outcome']],
    on='case_id',
    how='left'
)
always_wrong_export = always_wrong_export.merge(
    attention_per_case[['case_id', 'mean_attn_FACTS', 'mean_attn_APPLICANT', 'mean_attn_DEFENCE']],
    on='case_id',
    how='left'
)

# Add majority prediction
always_wrong_export = always_wrong_export.merge(
    majority_pred[['case_id', 'predicted']],
    on='case_id',
    how='left'
)
always_wrong_export['predicted_label'] = always_wrong_export['predicted'].map({0: 'Granted', 1: 'Refused'})
always_wrong_export['error_type'] = always_wrong_export.apply(
    lambda r: 'FP (pred Grant, was Refuse)' if r['label'] == 1 and r['predicted'] == 0 
              else 'FN (pred Refuse, was Grant)', axis=1
)

always_wrong_export.to_csv(DATA_DIR / "always_wrong_detailed.csv", index=False)
print(f"   ✅ Saved: always_wrong_detailed.csv ({len(always_wrong_export)} cases)")

# Summary stats
summary = {
    'total_always_wrong': len(always_wrong),
    'granted_always_wrong': int((always_wrong['label'] == 0).sum()),
    'refused_always_wrong': int((always_wrong['label'] == 1).sum()),
    'false_positives': fp,
    'false_negatives': fn,
}

import json
with open(DATA_DIR / "failure_analysis_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   ✅ Saved: failure_analysis_summary.json")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
print(f"\n   Key insight: {fp} False Positives, {fn} False Negatives")
print(f"   Check 'always_wrong_detailed.csv' to review specific cases")
