#!/usr/bin/env python3
"""
Verify all numerical claims in the report against the actual CSV data.
"""

import pandas as pd

# Load files
attention = pd.read_csv('attention_fixed_v2/attention_per_case.csv')
ablation = pd.read_csv('attention_fixed_v2/ablation_results.csv')

print('='*70)
print('VERIFICATION: Attention Density')
print('='*70)

aw = attention[attention['times_correct'] == 0]
ar = attention[attention['times_correct'] == 5]

print(f'Always Wrong (n={len(aw)}):')
print(f'  FACTS:     {aw["raw_density_FACTS"].mean():.4f}')
print(f'  APPLICANT: {aw["raw_density_APPLICANT"].mean():.4f}')
print(f'  DEFENCE:   {aw["raw_density_DEFENCE"].mean():.4f}')

print(f'Always Right (n={len(ar)}):')
print(f'  FACTS:     {ar["raw_density_FACTS"].mean():.4f}')
print(f'  APPLICANT: {ar["raw_density_APPLICANT"].mean():.4f}')
print(f'  DEFENCE:   {ar["raw_density_DEFENCE"].mean():.4f}')

print()
print('='*70)
print('VERIFICATION: Ablation by Consistency')
print('='*70)

for tc in sorted(ablation['times_correct'].unique()):
    subset = ablation[ablation['times_correct'] == tc]
    print(f'{int(tc)}/5 (n={len(subset)}): '
          f'FACTS={subset["facts_delta"].mean():+.4f}, '
          f'APPLICANT={subset["applicant_delta"].mean():+.4f}, '
          f'DEFENCE={subset["defence_delta"].mean():+.4f}')

print()
print('='*70)
print('VERIFICATION: FP vs FN Ablation (Always Wrong Only)')
print('='*70)

aw_abl = ablation[ablation['times_correct'] == 0]
fp = aw_abl[aw_abl['label'] == 1]  # True=Refused, Pred=Granted
fn = aw_abl[aw_abl['label'] == 0]  # True=Granted, Pred=Refused

print(f'False Positives (n={len(fp)}): DEFENCE delta = {fp["defence_delta"].mean():+.4f}')
print(f'False Negatives (n={len(fn)}): DEFENCE delta = {fn["defence_delta"].mean():+.4f}')

print()
print('='*70)
print('VERIFICATION: Flip Rates')
print('='*70)

for tc in [0, 5]:
    subset = ablation[ablation['times_correct'] == tc]
    label = "Always Wrong" if tc == 0 else "Always Right"
    print(f'{label} (n={len(subset)}): '
          f'FACTS={subset["facts_flipped"].mean():.1%}, '
          f'APPLICANT={subset["applicant_flipped"].mean():.1%}, '
          f'DEFENCE={subset["defence_flipped"].mean():.1%}')

print()
print('='*70)
print('REPORT CLAIMS vs ACTUAL DATA')
print('='*70)

# Check each claim from the report
claims = [
    # Attention density claims
    ('AW FACTS density', 0.0047, aw['raw_density_FACTS'].mean()),
    ('AW APPLICANT density', 0.0047, aw['raw_density_APPLICANT'].mean()),
    ('AW DEFENCE density', 0.0056, aw['raw_density_DEFENCE'].mean()),
    ('AR FACTS density', 0.0049, ar['raw_density_FACTS'].mean()),
    ('AR APPLICANT density', 0.0049, ar['raw_density_APPLICANT'].mean()),
    ('AR DEFENCE density', 0.0059, ar['raw_density_DEFENCE'].mean()),
    
    # Ablation delta claims
    ('FP DEFENCE delta', +0.1289, fp['defence_delta'].mean()),
    ('FN DEFENCE delta', -0.2489, fn['defence_delta'].mean()),
    
    # Ablation by consistency (0/5)
    ('0/5 FACTS delta', +0.1579, ablation[ablation['times_correct']==0]['facts_delta'].mean()),
    ('0/5 APPLICANT delta', +0.1151, ablation[ablation['times_correct']==0]['applicant_delta'].mean()),
    ('0/5 DEFENCE delta', +0.0250, ablation[ablation['times_correct']==0]['defence_delta'].mean()),
    
    # Ablation by consistency (5/5)
    ('5/5 FACTS delta', +0.1639, ablation[ablation['times_correct']==5]['facts_delta'].mean()),
    ('5/5 APPLICANT delta', +0.0848, ablation[ablation['times_correct']==5]['applicant_delta'].mean()),
    ('5/5 DEFENCE delta', -0.0196, ablation[ablation['times_correct']==5]['defence_delta'].mean()),
]

all_match = True
for name, claimed, actual in claims:
    match = abs(claimed - actual) < 0.001
    symbol = '✅' if match else '❌'
    if not match:
        all_match = False
    print(f'{name}: claimed={claimed:+.4f}, actual={actual:+.4f} {symbol}')

print()
print('='*70)
print('SUMMARY')
print('='*70)

if all_match:
    print('✅ ALL CLAIMS VERIFIED - Numbers in report match CSV data')
else:
    print('❌ SOME CLAIMS DO NOT MATCH - Review discrepancies above')

# Additional counts verification
print()
print('='*70)
print('ADDITIONAL COUNTS')
print('='*70)
print(f'Total cases in attention_per_case.csv: {len(attention)}')
print(f'Always Wrong (0/5): {len(aw)}')
print(f'Always Right (5/5): {len(ar)}')
print(f'Total ablation samples: {len(ablation)}')
print(f'Ablation FP (in 0/5): {len(fp)}')
print(f'Ablation FN (in 0/5): {len(fn)}')
