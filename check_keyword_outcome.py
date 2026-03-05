#!/usr/bin/env python3
"""
Decisive check: Error rate by keyword, SPLIT BY OUTCOME.

If binding/defamatory increases error mainly in REFUSED cases:
→ Model predicts GRANTED when it should predict REFUSED in these domains
→ Perfectly aligns with the 110 false positives finding
"""

import pandas as pd

# Load data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))

consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')
df = df.merge(consistency[['case_id', 'times_correct']], on='case_id')

df['all_text'] = (df['facts'].fillna('') + ' ' + 
                  df['applicant_reason'].fillna('') + ' ' + 
                  df['defence_reason'].fillna('')).str.lower()

# Split by outcome
granted = df[df['outcome'] == 'summary judgment granted']
refused = df[df['outcome'] == 'summary judgment refused']

print("="*70)
print("BASELINE ERROR RATES BY OUTCOME")
print("="*70)
granted_err = (granted['times_correct'] == 0).mean() * 100
refused_err = (refused['times_correct'] == 0).mean() * 100
print(f"GRANTED cases (n={len(granted)}): {granted_err:.1f}% always-wrong")
print(f"REFUSED cases (n={len(refused)}): {refused_err:.1f}% always-wrong")

print("\n" + "="*70)
print("ERROR RATE BY KEYWORD × OUTCOME")
print("="*70)

keywords = ['binding', 'defamatory', 'settlement', 'aside', 'guarantee', 'property']

print(f"\n{'Keyword':<15} {'GRANTED':<20} {'REFUSED':<20} {'REFUSED lift':<15}")
print(f"{'':15} {'Err% (n)':<20} {'Err% (n)':<20} {'vs baseline':<15}")
print("-"*70)

for word in keywords:
    # GRANTED cases with/without keyword
    g_has = granted['all_text'].str.contains(word)
    g_err_with = (granted[g_has]['times_correct'] == 0).mean() * 100
    g_n = g_has.sum()
    
    # REFUSED cases with/without keyword
    r_has = refused['all_text'].str.contains(word)
    r_err_with = (refused[r_has]['times_correct'] == 0).mean() * 100
    r_n = r_has.sum()
    
    # Lift vs baseline
    r_lift = r_err_with / refused_err if refused_err > 0 else 0
    
    print(f"{word:<15} {g_err_with:>5.1f}% ({g_n:<4})      {r_err_with:>5.1f}% ({r_n:<4})      {r_lift:.2f}x")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
If REFUSED error rate spikes (e.g., 25%+) while GRANTED stays low:
  → Model predicts GRANTED when it should predict REFUSED in these domains
  → This is exactly the False Positive pattern (110 of 153 always-wrong)
  
If BOTH spike equally:
  → Domain is just hard regardless of direction
""")

# Detailed breakdown for top keywords
print("\n" + "="*70)
print("DETAILED: Cases with 'binding'")
print("="*70)
binding_cases = df[df['all_text'].str.contains('binding')]
binding_wrong = binding_cases[binding_cases['times_correct'] == 0]

print(f"Total 'binding' cases: {len(binding_cases)}")
print(f"Always wrong: {len(binding_wrong)}")
print(f"\nOf the {len(binding_wrong)} always-wrong 'binding' cases:")
print(f"  - True GRANTED (false negatives): {(binding_wrong['outcome'] == 'summary judgment granted').sum()}")
print(f"  - True REFUSED (false positives): {(binding_wrong['outcome'] == 'summary judgment refused').sum()}")

print("\n" + "="*70)
print("DETAILED: Cases with 'defamatory'")
print("="*70)
defam_cases = df[df['all_text'].str.contains('defamatory')]
defam_wrong = defam_cases[defam_cases['times_correct'] == 0]

print(f"Total 'defamatory' cases: {len(defam_cases)}")
print(f"Always wrong: {len(defam_wrong)}")
print(f"\nOf the {len(defam_wrong)} always-wrong 'defamatory' cases:")
print(f"  - True GRANTED (false negatives): {(defam_wrong['outcome'] == 'summary judgment granted').sum()}")
print(f"  - True REFUSED (false positives): {(defam_wrong['outcome'] == 'summary judgment refused').sum()}")
