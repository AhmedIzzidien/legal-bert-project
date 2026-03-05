#!/usr/bin/env python3
"""
Check if Always-Wrong cases are concentrated in specific categories.
"""

import pandas as pd

# Load data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))

consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')
df = df.merge(consistency[['case_id', 'times_correct', 'stratum']], on='case_id')

print("="*70)
print("ERROR RATE BY STRATUM (L/E/T)")
print("="*70)

# Error rate by stratum
stratum_stats = df.groupby('stratum').agg(
    total=('case_id', 'count'),
    always_wrong=('times_correct', lambda x: (x==0).sum()),
    always_right=('times_correct', lambda x: (x==5).sum())
).reset_index()

stratum_stats['error_rate'] = stratum_stats['always_wrong'] / stratum_stats['total'] * 100
stratum_stats['success_rate'] = stratum_stats['always_right'] / stratum_stats['total'] * 100
stratum_stats = stratum_stats.sort_values('error_rate', ascending=False)

print(f"\n{'Stratum':<25} {'Total':<8} {'Wrong':<8} {'Right':<8} {'Err%':<8} {'Succ%':<8}")
print("-"*65)
for _, row in stratum_stats.iterrows():
    print(f"{row['stratum']:<25} {row['total']:<8} {row['always_wrong']:<8} {row['always_right']:<8} {row['error_rate']:<8.1f} {row['success_rate']:<8.1f}")

# Overall stats
print("\n" + "="*70)
print("ERROR RATE BY OUTCOME")
print("="*70)

outcome_stats = df.groupby('outcome').agg(
    total=('case_id', 'count'),
    always_wrong=('times_correct', lambda x: (x==0).sum()),
    always_right=('times_correct', lambda x: (x==5).sum())
).reset_index()

outcome_stats['error_rate'] = outcome_stats['always_wrong'] / outcome_stats['total'] * 100
outcome_stats['success_rate'] = outcome_stats['always_right'] / outcome_stats['total'] * 100

print(f"\n{'Outcome':<30} {'Total':<8} {'Wrong':<8} {'Right':<8} {'Err%':<8} {'Succ%':<8}")
print("-"*70)
for _, row in outcome_stats.iterrows():
    print(f"{row['outcome']:<30} {row['total']:<8} {row['always_wrong']:<8} {row['always_right']:<8} {row['error_rate']:<8.1f} {row['success_rate']:<8.1f}")

# Check decision_reason_categories if useful
print("\n" + "="*70)
print("SAMPLE decision_reason_categories_clean VALUES")
print("="*70)
print(df['decision_reason_categories_clean'].value_counts().head(15))

# Overall
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
overall_error = df['times_correct'].eq(0).sum() / len(df) * 100
overall_success = df['times_correct'].eq(5).sum() / len(df) * 100
print(f"\nOverall error rate (0/5): {overall_error:.1f}%")
print(f"Overall success rate (5/5): {overall_success:.1f}%")
