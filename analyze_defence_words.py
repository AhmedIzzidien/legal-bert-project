#!/usr/bin/env python3
"""
Analyze which words in DEFENCE section are over-represented in always-wrong cases.
"""

import pandas as pd
import re
from collections import Counter

# Load data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))

consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')
df = df.merge(consistency[['case_id', 'times_correct']], on='case_id')

# Split groups
always_wrong = df[df['times_correct'] == 0]['defence_reason'].fillna('').tolist()
always_right = df[df['times_correct'] == 5]['defence_reason'].fillna('').tolist()

print(f"Always Wrong: {len(always_wrong)} cases")
print(f"Always Right: {len(always_right)} cases")

# Word frequency function
def get_words(texts):
    words = []
    for t in texts:
        words.extend(re.findall(r'\b[a-z]{4,}\b', t.lower()))
    return Counter(words)

wrong_words = get_words(always_wrong)
right_words = get_words(always_right)

# Find over-represented words in failures
print("\n" + "="*60)
print("DEFENCE WORDS OVER-REPRESENTED IN ALWAYS-WRONG")
print("="*60)
print(f"{'Word':<20} {'Wrong %':<12} {'Right %':<12} {'Ratio':<8}")
print("-"*52)

results = []
for word, count in wrong_words.most_common(500):
    wrong_rate = count / len(always_wrong)
    right_count = right_words.get(word, 0)
    right_rate = right_count / len(always_right)
    
    if wrong_rate > 0.05 and right_rate > 0.01 and wrong_rate / right_rate > 1.5:
        results.append((word, wrong_rate, right_rate, wrong_rate/right_rate))

# Sort by ratio
results.sort(key=lambda x: x[3], reverse=True)

for word, wrong_rate, right_rate, ratio in results[:30]:
    print(f"{word:<20} {wrong_rate:>6.1%}       {right_rate:>6.1%}       {ratio:.1f}x")

# Also show words UNDER-represented in failures (more common in successes)
print("\n" + "="*60)
print("DEFENCE WORDS UNDER-REPRESENTED IN ALWAYS-WRONG")
print("(More common in successful predictions)")
print("="*60)
print(f"{'Word':<20} {'Wrong %':<12} {'Right %':<12} {'Ratio':<8}")
print("-"*52)

under_results = []
for word, count in right_words.most_common(500):
    right_rate = count / len(always_right)
    wrong_count = wrong_words.get(word, 0)
    wrong_rate = wrong_count / len(always_wrong)
    
    if right_rate > 0.05 and wrong_rate > 0.01 and right_rate / wrong_rate > 1.5:
        under_results.append((word, wrong_rate, right_rate, right_rate/wrong_rate))

under_results.sort(key=lambda x: x[3], reverse=True)

for word, wrong_rate, right_rate, ratio in under_results[:30]:
    print(f"{word:<20} {wrong_rate:>6.1%}       {right_rate:>6.1%}       {ratio:.1f}x")
