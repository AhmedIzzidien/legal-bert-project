#!/usr/bin/env python3
"""
Check if cases containing specific words have higher error rates.

This tests whether the vocabulary patterns (defamatory, binding, etc.)
are associated with higher failure rates.
"""

import pandas as pd

# Load data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))

consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')
df = df.merge(consistency[['case_id', 'times_correct']], on='case_id')

# Combine all text
df['all_text'] = (df['facts'].fillna('') + ' ' + 
                  df['applicant_reason'].fillna('') + ' ' + 
                  df['defence_reason'].fillna('')).str.lower()

overall_error = (df['times_correct'] == 0).mean() * 100

print("="*70)
print("ERROR RATE BY KEYWORD PRESENCE")
print("="*70)
print(f"\nOverall error rate: {overall_error:.1f}%")
print(f"\n{'Keyword':<15} {'N':<8} {'Err% WITH':<12} {'Err% WITHOUT':<14} {'Ratio':<8}")
print("-"*60)

# Keywords from defence word analysis
keywords = ['defamatory', 'binding', 'guarantee', 'property', 'settlement', 
            'aside', 'payment', 'first', 'regarding']

results = []
for word in keywords:
    has_word = df['all_text'].str.contains(word)
    n_with = has_word.sum()
    
    err_with = (df[has_word]['times_correct'] == 0).mean() * 100
    err_without = (df[~has_word]['times_correct'] == 0).mean() * 100
    
    ratio = err_with / err_without if err_without > 0 else 0
    results.append((word, n_with, err_with, err_without, ratio))

# Sort by ratio
results.sort(key=lambda x: x[4], reverse=True)

for word, n, err_with, err_without, ratio in results:
    print(f"{word:<15} {n:<8} {err_with:<12.1f} {err_without:<14.1f} {ratio:<8.2f}x")

# Also check the "good" words (under-represented in failures)
print("\n" + "="*70)
print("WORDS ASSOCIATED WITH SUCCESS")
print("="*70)
print(f"\n{'Keyword':<15} {'N':<8} {'Err% WITH':<12} {'Err% WITHOUT':<14} {'Ratio':<8}")
print("-"*60)

good_keywords = ['claimant', 'defendants', 'court', 'sought', 'dispute', 
                 'enforcement', 'prospect', 'after', 'also']

good_results = []
for word in good_keywords:
    has_word = df['all_text'].str.contains(word)
    n_with = has_word.sum()
    
    err_with = (df[has_word]['times_correct'] == 0).mean() * 100
    err_without = (df[~has_word]['times_correct'] == 0).mean() * 100
    
    ratio = err_with / err_without if err_without > 0 else 0
    good_results.append((word, n_with, err_with, err_without, ratio))

# Sort by ratio (ascending for good words)
good_results.sort(key=lambda x: x[4])

for word, n, err_with, err_without, ratio in good_results:
    print(f"{word:<15} {n:<8} {err_with:<12.1f} {err_without:<14.1f} {ratio:<8.2f}x")

# Summary
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
If 'defamatory' cases have 15%+ error rate vs 7% overall:
  → Domain imbalance is driving failures (specific case types are hard)

If error rates are similar regardless of keyword:
  → The lexical pattern reflects something else (not case type difficulty)
""")
