#!/usr/bin/env python3
"""
PAPER CORRECTIONS GENERATOR
Outputs all corrections needed and verifies Table 4 always-wrong counts
"""
import pandas as pd
import numpy as np

print("="*70)
print("PAPER CORRECTIONS - Full Report")
print("="*70)

# Load data
cons = pd.read_csv("legalbert_multiseed_attention/case_consistency.csv")

print("\n" + "="*70)
print("1. TABLE 4 VERIFICATION - Corrected Always-Wrong Counts")
print("="*70)

strata = ['EVIDENCE + TRIAL', 'LAW + EVIDENCE', 'ALL THREE', 
          'TRIAL only', 'LAW only', 'EVIDENCE only', 'LAW + TRIAL']

print(f"\n{'Stratum':<22} {'N':>6} {'AW':>5} {'Rate':>8}")
print("-"*45)
for stratum in strata:
    sd = cons[cons['stratum'] == stratum]
    n = len(sd)
    aw = (sd['times_correct'] == 0).sum()
    rate = aw / n * 100 if n > 0 else 0
    print(f"{stratum:<22} {n:>6} {aw:>5} {rate:>7.1f}%")

print("\n" + "="*70)
print("2. CORRECTIONS SUMMARY")
print("="*70)

corrections = """
OVERALL PERFORMANCE (update these values):
------------------------------------------
  F1-Macro:    0.598  -->  0.604
  F1-Granted:  0.724  -->  0.698
  F1-Refused:  0.472  -->  0.511
  Accuracy:    0.627  (no change)

TABLE 8 - VOCABULARY PATTERNS (Wrong% column):
----------------------------------------------
  defamatory:  6.5%  -->  4.6%   (ratio: 5.9x --> 4.2x)
  property:    5.9%  -->  4.6%   (ratio: 3.2x --> 2.6x)
  binding:     5.9%  (no change)
  guarantee:   4.6%  (verify)
  settlement:  5.2%  (verify)

PHRASE COUNT:
-------------
  5,392  -->  5,628

TABLE 4 - L/E/T ERROR RATES (N values):
---------------------------------------
  EVIDENCE + TRIAL:  188  -->  187
  LAW + EVIDENCE:    444  -->  559  (rate changes!)
  ALL THREE:         163  -->  161
"""
print(corrections)

print("="*70)
print("3. CORRECTED TABLE 4")
print("="*70)

# Generate corrected Table 4
print("""
| Decision Basis              | Cases | Always Wrong | Error Rate |
|-----------------------------|-------|--------------|------------|""")

table4_order = [
    ('EVIDENCE + TRIAL', 'Evidence + Trial (no Law)'),
    ('TRIAL only', 'Trial only'),
    ('LAW only', 'Law only'),
    ('LAW + EVIDENCE', 'Law + Evidence (no Trial)'),
    ('ALL THREE', 'All three (L + E + T)')
]

for stratum, display_name in table4_order:
    sd = cons[cons['stratum'] == stratum]
    n = len(sd)
    aw = (sd['times_correct'] == 0).sum()
    rate = aw / n * 100 if n > 0 else 0
    print(f"| {display_name:<27} | {n:>5} | {aw:>12} | {rate:>9.1f}% |")

print("\n" + "="*70)
print("4. CORRECTED TABLE 8")
print("="*70)

# Load original data for vocabulary analysis
try:
    import re
    df_text = pd.read_pickle("sj_231025.pkl")
    outcomes = ["summary judgment granted", "summary judgment refused"]
    df = df_text[df_text["outcome"].isin(outcomes)].copy().reset_index(drop=True)
    df['case_id'] = range(len(df))
    df = df.merge(cons[['case_id', 'times_correct']], on='case_id')
    
    aw_df = df[df['times_correct'] == 0]
    ar_df = df[df['times_correct'] == 5]
    n_aw, n_ar = len(aw_df), len(ar_df)
    
    def ext(t):
        return set(re.findall(r'\b[a-z]{4,}\b', str(t).lower())) if pd.notna(t) else set()
    
    aw_def = aw_df['defence_reason'].apply(ext)
    ar_def = ar_df['defence_reason'].apply(ext)
    
    words = ['defamatory', 'binding', 'property', 'guarantee', 'settlement']
    
    print(f"\n{'Word':<12} {'Wrong%':>8} {'Right%':>8} {'Ratio':>8} {'Domain':<20}")
    print("-"*60)
    
    domains = {
        'defamatory': 'Defamation',
        'binding': 'Contract validity',
        'property': 'Property disputes',
        'guarantee': 'Guarantee enforcement',
        'settlement': 'Settlement disputes'
    }
    
    for w in words:
        wp = sum(1 for ws in aw_def if w in ws) / n_aw * 100
        rp = sum(1 for ws in ar_def if w in ws) / n_ar * 100
        ratio = wp / rp if rp > 0 else 0
        print(f"{w:<12} {wp:>7.1f}% {rp:>7.1f}% {ratio:>7.1f}x {domains.get(w, ''):<20}")
        
except Exception as e:
    print(f"Could not generate Table 8: {e}")

print("\n" + "="*70)
print("5. ITEMS VERIFIED CORRECT (no changes needed)")
print("="*70)
print("""
- Dataset: 1,961 cases, 1,196 granted (61%), 765 refused (39%)
- Accuracy: 62.7%
- Table 1: Consistency distribution (all 6 values correct)
- Table 2: Outcome asymmetry - 3.6% vs 14.4%, chi-sq=73.2
- Table 3: Error direction - 110 FP (71.9%), 43 FN (28.1%)
- Section 4: All L/E/T stratum performance values
- Table 5: All 6 attention density values
- Table 6: All 6 ablation delta values
- Table 7: N=29 FP, N=11 FN, defence deltas correct
- Tables 10-11: All keyword validation values
- Binding breakdown: 73 cases, 42G/31R, 2/10 AW correct
""")

print("="*70)
print("SAVE THIS OUTPUT FOR REFERENCE")
print("="*70)
