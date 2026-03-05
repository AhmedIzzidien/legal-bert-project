#!/usr/bin/env python3
"""
===============================================================================
MULTI-SEED TOPIC ANALYSIS (FIXED)
===============================================================================

Merges multi-seed consistency results with topic labels to produce
ROBUST topic difficulty rankings based on 5-seed × 5-fold evaluation.

Inputs:
    - case_consistency.csv (from multi-seed run) - has 'times_correct' column
    - sj_231025_w_topics_all_cases.pkl (with topic labels)

Outputs:
    - Topic error rates based on multi-seed consistency
    - Topic × L/E/T cross-analysis
    - Publication-ready tables and plots

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# CONFIGURATION
# ===============================================================================

CONSISTENCY_FILE = "legalbert_multiseed_attention/case_consistency.csv"
TOPICS_FILE = "sj_231025_w_topics_all_cases.pkl"
OUTPUT_DIR = "multiseed_topic_analysis"

# Expected counts (sanity check)
EXPECTED_ALWAYS_WRONG = 153
EXPECTED_ALWAYS_RIGHT = 546

# ===============================================================================
# LOAD AND MERGE DATA
# ===============================================================================

print("="*70)
print("📊 MULTI-SEED TOPIC ANALYSIS")
print("="*70)

# Load consistency data
print(f"\n📂 Loading: {CONSISTENCY_FILE}")
df_consistency = pd.read_csv(CONSISTENCY_FILE)
print(f"   Rows: {len(df_consistency)}")
print(f"   Columns: {list(df_consistency.columns)}")

# Verify times_correct column exists
if 'times_correct' not in df_consistency.columns:
    print(f"\n❌ ERROR: 'times_correct' column not found!")
    exit(1)

# Load topics data
print(f"\n📂 Loading: {TOPICS_FILE}")
df_topics = pd.read_pickle(TOPICS_FILE)
print(f"   Rows: {len(df_topics)}")

# Check alignment
if len(df_consistency) != len(df_topics):
    print(f"\n❌ ERROR: Row counts don't match!")
    exit(1)

# Merge on index (verified aligned via text match earlier)
print(f"\n🔗 Merging on index...")
df = df_consistency.copy()
df['primary_topic'] = df_topics['primary_topic'].values
df['secondary_topic'] = df_topics['secondary_topic'].values
df['outcome'] = df_topics['outcome'].values
df['decision_reason_categories'] = df_topics['decision_reason_categories'].values

# Parse L/E/T from decision_reason_categories
def parse_let(cat):
    """Parse Law/Evidence/Trial from decision_reason_categories."""
    if pd.isna(cat):
        return 0, 0, 0
    cat = str(cat).upper()
    L = 1 if 'LAW' in cat else 0
    E = 1 if 'EVIDENCE' in cat else 0
    T = 1 if 'TRIAL' in cat else 0
    return L, E, T

df[['L', 'E', 'T']] = df['decision_reason_categories'].apply(lambda x: pd.Series(parse_let(x)))
df['LET_sum'] = df['L'] + df['E'] + df['T']
df['LET_strata'] = df.apply(lambda r: f"L={r['L']} E={r['E']} T={r['T']}", axis=1)

# Use times_correct as consistency measure
consistency_col = 'times_correct'
print(f"   Using consistency column: '{consistency_col}'")

# Define always-wrong and always-right
df['always_wrong'] = df[consistency_col] == 0
df['always_right'] = df[consistency_col] == 5
df['structurally_hard'] = df[consistency_col] <= 1  # 0/5 or 1/5

print(f"   ✅ Merged: {len(df)} cases")

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ===============================================================================
# FILTER TO BINARY CASES
# ===============================================================================

print("\n" + "="*70)
print("📊 FILTERING TO BINARY CASES")
print("="*70)

print(f"\n   Raw outcome distribution:")
print(df['outcome'].value_counts())

# Normalize outcomes (handle full string format)
def normalize_outcome(outcome):
    if pd.isna(outcome):
        return None
    outcome = str(outcome).lower().strip()
    if 'granted' in outcome and 'refused' not in outcome and 'partly' not in outcome:
        return 'GRANTED'
    elif 'refused' in outcome:
        return 'REFUSED'
    else:
        return None

df['outcome_binary'] = df['outcome'].apply(normalize_outcome)

# Filter to binary cases
df_binary = df[df['outcome_binary'].isin(['GRANTED', 'REFUSED'])].copy()
print(f"\n   Binary cases: {len(df_binary)} / {len(df)}")
print(f"   GRANTED: {(df_binary['outcome_binary'] == 'GRANTED').sum()}")
print(f"   REFUSED: {(df_binary['outcome_binary'] == 'REFUSED').sum()}")

# ===============================================================================
# SANITY CHECK COUNTS
# ===============================================================================

print("\n" + "="*70)
print("📊 SANITY CHECK")
print("="*70)

n_always_wrong = df_binary['always_wrong'].sum()
n_always_right = df_binary['always_right'].sum()

print(f"\n   Always-wrong (0/5): {n_always_wrong} cases")
print(f"   Always-right (5/5): {n_always_right} cases")

if abs(n_always_wrong - EXPECTED_ALWAYS_WRONG) > 5:
    print(f"   ⚠️  Expected ~{EXPECTED_ALWAYS_WRONG} always-wrong, got {n_always_wrong}")
else:
    print(f"   ✅ Always-wrong count matches expected")

if abs(n_always_right - EXPECTED_ALWAYS_RIGHT) > 10:
    print(f"   ⚠️  Expected ~{EXPECTED_ALWAYS_RIGHT} always-right, got {n_always_right}")
else:
    print(f"   ✅ Always-right count matches expected")

# ===============================================================================
# CONSISTENCY DISTRIBUTION
# ===============================================================================

print("\n" + "="*70)
print("📊 CONSISTENCY DISTRIBUTION")
print("="*70)

print(f"\n   Distribution of times_correct (0-5):")
print(df_binary[consistency_col].value_counts().sort_index())

# ===============================================================================
# TOPIC ERROR RATE ANALYSIS (MULTI-SEED)
# ===============================================================================

print("\n" + "="*70)
print("📊 TOPIC ERROR RATES (MULTI-SEED ROBUST)")
print("="*70)

def analyze_topic_errors(df, topic_col, min_cases=10):
    """Compute error rates by topic using multi-seed consistency."""
    
    results = []
    
    for topic in df[topic_col].unique():
        if pd.isna(topic):
            continue
            
        topic_df = df[df[topic_col] == topic]
        n = len(topic_df)
        
        if n < min_cases:
            continue
        
        # Count always-wrong (0/5 consistency)
        n_always_wrong = topic_df['always_wrong'].sum()
        
        # Count structurally hard (0/5 or 1/5)
        n_hard = topic_df['structurally_hard'].sum()
        
        # Count always-right (5/5 consistency)
        n_always_right = topic_df['always_right'].sum()
        
        # Mean consistency
        mean_consistency = topic_df[consistency_col].mean()
        
        results.append({
            'topic': topic,
            'n_cases': n,
            'n_always_wrong': n_always_wrong,
            'always_wrong_rate': n_always_wrong / n,
            'n_structurally_hard': n_hard,
            'hard_rate': n_hard / n,
            'n_always_right': n_always_right,
            'always_right_rate': n_always_right / n,
            'mean_consistency': mean_consistency
        })
    
    return pd.DataFrame(results).sort_values('always_wrong_rate', ascending=False)

# Analyze primary topics
topic_stats = analyze_topic_errors(df_binary, 'primary_topic', min_cases=20)

print(f"\n{'Topic':<45} {'N':>6} {'Always Wrong':>14} {'Hard (0-1/5)':>14} {'Always Right':>14} {'Mean Cons':>10}")
print("-"*105)

for _, row in topic_stats.head(15).iterrows():
    print(f"{row['topic']:<45} {row['n_cases']:>6} "
          f"{row['n_always_wrong']:>6} ({row['always_wrong_rate']*100:>5.1f}%) "
          f"{row['n_structurally_hard']:>6} ({row['hard_rate']*100:>5.1f}%) "
          f"{row['n_always_right']:>6} ({row['always_right_rate']*100:>5.1f}%) "
          f"{row['mean_consistency']:>10.2f}")

# Save
topic_stats.to_csv(f"{OUTPUT_DIR}/topic_error_rates_multiseed.csv", index=False)
print(f"\n   ✅ Saved: topic_error_rates_multiseed.csv")

# ===============================================================================
# TOPIC × OUTCOME INTERACTION (KEY FOR PAPER)
# ===============================================================================

print("\n" + "="*70)
print("📊 TOPIC × OUTCOME ERROR RATES")
print("="*70)

def analyze_topic_outcome_interaction(df, topic_col, min_cases=10):
    """Check if error rates differ by outcome within topics."""
    
    results = []
    
    for topic in df[topic_col].unique():
        if pd.isna(topic):
            continue
            
        topic_df = df[df[topic_col] == topic]
        
        for outcome in ['GRANTED', 'REFUSED']:
            outcome_df = topic_df[topic_df['outcome_binary'] == outcome]
            n = len(outcome_df)
            
            if n < 5:
                continue
            
            n_always_wrong = outcome_df['always_wrong'].sum()
            
            results.append({
                'topic': topic,
                'outcome': outcome,
                'n_cases': n,
                'n_always_wrong': n_always_wrong,
                'error_rate': n_always_wrong / n if n > 0 else 0
            })
    
    return pd.DataFrame(results)

topic_outcome = analyze_topic_outcome_interaction(df_binary, 'primary_topic')

# Pivot for easy comparison
if len(topic_outcome) > 0:
    pivot = topic_outcome.pivot_table(
        index='topic', 
        columns='outcome', 
        values='error_rate',
        aggfunc='first'
    ).reset_index()
    
    if 'GRANTED' in pivot.columns and 'REFUSED' in pivot.columns:
        pivot['ratio'] = pivot['REFUSED'] / pivot['GRANTED'].replace(0, np.nan)
        pivot = pivot.sort_values('REFUSED', ascending=False)
        
        print(f"\n{'Topic':<40} {'GRANTED err%':>14} {'REFUSED err%':>14} {'Ratio':>10}")
        print("-"*80)
        
        for _, row in pivot.head(10).iterrows():
            g = row.get('GRANTED', 0) * 100
            r = row.get('REFUSED', 0) * 100
            ratio = row.get('ratio', np.nan)
            ratio_str = f"{ratio:.1f}×" if pd.notna(ratio) and not np.isinf(ratio) else "—"
            print(f"{row['topic']:<40} {g:>13.1f}% {r:>13.1f}% {ratio_str:>10}")
        
        pivot.to_csv(f"{OUTPUT_DIR}/topic_x_outcome_error_rates.csv", index=False)
        print(f"\n   ✅ Saved: topic_x_outcome_error_rates.csv")

# ===============================================================================
# L/E/T ANALYSIS (MULTI-SEED)
# ===============================================================================

print("\n" + "="*70)
print("📊 L/E/T STRATIFIED ERROR RATES (MULTI-SEED)")
print("="*70)

let_stats = []
for strata in df_binary['LET_strata'].unique():
    strata_df = df_binary[df_binary['LET_strata'] == strata]
    n = len(strata_df)
    
    if n < 10:
        continue
    
    let_stats.append({
        'strata': strata,
        'n_cases': n,
        'n_always_wrong': strata_df['always_wrong'].sum(),
        'always_wrong_rate': strata_df['always_wrong'].mean(),
        'n_always_right': strata_df['always_right'].sum(),
        'always_right_rate': strata_df['always_right'].mean(),
        'mean_consistency': strata_df[consistency_col].mean()
    })

let_df = pd.DataFrame(let_stats).sort_values('always_wrong_rate', ascending=False)

print(f"\n{'L/E/T Strata':<20} {'N':>8} {'Always Wrong':>16} {'Always Right':>16} {'Mean Cons':>12}")
print("-"*75)

for _, row in let_df.iterrows():
    print(f"{row['strata']:<20} {row['n_cases']:>8} "
          f"{row['n_always_wrong']:>6} ({row['always_wrong_rate']*100:>5.1f}%) "
          f"{row['n_always_right']:>6} ({row['always_right_rate']*100:>5.1f}%) "
          f"{row['mean_consistency']:>12.2f}")

let_df.to_csv(f"{OUTPUT_DIR}/let_error_rates_multiseed.csv", index=False)
print(f"\n   ✅ Saved: let_error_rates_multiseed.csv")

# ===============================================================================
# TOPIC × L/E/T CROSS-ANALYSIS
# ===============================================================================

print("\n" + "="*70)
print("📊 TOPIC × L/E/T CROSS-ANALYSIS (TOP COMBINATIONS)")
print("="*70)

cross_stats = []
for topic in df_binary['primary_topic'].unique():
    if pd.isna(topic):
        continue
    for strata in df_binary['LET_strata'].unique():
        subset = df_binary[(df_binary['primary_topic'] == topic) & (df_binary['LET_strata'] == strata)]
        n = len(subset)
        
        if n < 10:
            continue
        
        cross_stats.append({
            'topic': topic,
            'LET_strata': strata,
            'n_cases': n,
            'always_wrong_rate': subset['always_wrong'].mean(),
            'always_right_rate': subset['always_right'].mean(),
            'mean_consistency': subset[consistency_col].mean()
        })

cross_df = pd.DataFrame(cross_stats).sort_values('mean_consistency', ascending=False)

print(f"\n{'Topic':<35} {'L/E/T':<18} {'N':>6} {'Err%':>8} {'Right%':>8} {'Cons':>8}")
print("-"*90)

for _, row in cross_df.head(10).iterrows():
    print(f"{row['topic']:<35} {row['LET_strata']:<18} {row['n_cases']:>6} "
          f"{row['always_wrong_rate']*100:>7.1f}% {row['always_right_rate']*100:>7.1f}% "
          f"{row['mean_consistency']:>8.2f}")

print("\n... WORST combinations:")
for _, row in cross_df.tail(5).iterrows():
    print(f"{row['topic']:<35} {row['LET_strata']:<18} {row['n_cases']:>6} "
          f"{row['always_wrong_rate']*100:>7.1f}% {row['always_right_rate']*100:>7.1f}% "
          f"{row['mean_consistency']:>8.2f}")

cross_df.to_csv(f"{OUTPUT_DIR}/topic_x_let_multiseed.csv", index=False)
print(f"\n   ✅ Saved: topic_x_let_multiseed.csv")

# ===============================================================================
# VISUALIZATION
# ===============================================================================

print("\n" + "="*70)
print("📊 GENERATING PLOTS")
print("="*70)

# Plot 1: Topic error rates
fig, ax = plt.subplots(figsize=(12, 8))
top_topics = topic_stats.head(15)
colors = plt.cm.RdYlGn_r(top_topics['always_wrong_rate'] / top_topics['always_wrong_rate'].max())
bars = ax.barh(range(len(top_topics)), top_topics['always_wrong_rate'] * 100, color=colors)
ax.set_yticks(range(len(top_topics)))
ax.set_yticklabels(top_topics['topic'])
ax.set_xlabel('Always-Wrong Rate (%)')
ax.set_title('Topic Difficulty (Multi-Seed: 0/5 Consistency = Structurally Hard)')
ax.invert_yaxis()

for i, (rate, n) in enumerate(zip(top_topics['always_wrong_rate'], top_topics['n_cases'])):
    ax.text(rate * 100 + 0.5, i, f'n={n}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/topic_error_rates_multiseed.png", dpi=150)
print(f"   ✅ Saved: topic_error_rates_multiseed.png")

# Plot 2: L/E/T error rates
fig, ax = plt.subplots(figsize=(10, 6))
let_sorted = let_df.sort_values('always_wrong_rate', ascending=True)
colors = plt.cm.RdYlGn_r(let_sorted['always_wrong_rate'] / let_sorted['always_wrong_rate'].max())
bars = ax.barh(range(len(let_sorted)), let_sorted['always_wrong_rate'] * 100, color=colors)
ax.set_yticks(range(len(let_sorted)))
ax.set_yticklabels(let_sorted['strata'])
ax.set_xlabel('Always-Wrong Rate (%)')
ax.set_title('L/E/T Stratum Difficulty (Multi-Seed)')

for i, (rate, n) in enumerate(zip(let_sorted['always_wrong_rate'], let_sorted['n_cases'])):
    ax.text(rate * 100 + 0.3, i, f'n={n}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/let_error_rates_multiseed.png", dpi=150)
print(f"   ✅ Saved: let_error_rates_multiseed.png")

# Plot 3: Consistency distribution by topic (heatmap)
if len(topic_stats) > 5:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create consistency distribution per topic
    top_n = 12
    top_topics_list = topic_stats.head(top_n)['topic'].tolist()
    
    consistency_dist = []
    for topic in top_topics_list:
        topic_df = df_binary[df_binary['primary_topic'] == topic]
        dist = topic_df[consistency_col].value_counts(normalize=True).sort_index()
        consistency_dist.append({
            'topic': topic,
            **{f'{i}/5': dist.get(i, 0) for i in range(6)}
        })
    
    dist_df = pd.DataFrame(consistency_dist).set_index('topic')
    
    # Use matplotlib for heatmap instead of seaborn
    im = ax.imshow(dist_df.values, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(range(len(dist_df.columns)))
    ax.set_xticklabels(dist_df.columns)
    ax.set_yticks(range(len(dist_df.index)))
    ax.set_yticklabels(dist_df.index)
    
    # Add text annotations
    for i in range(len(dist_df.index)):
        for j in range(len(dist_df.columns)):
            val = dist_df.iloc[i, j]
            ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=8)
    
    ax.set_title('Consistency Distribution by Topic (Multi-Seed)')
    ax.set_xlabel('Consistency (correct seeds / 5)')
    
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/topic_consistency_heatmap.png", dpi=150)
    print(f"   ✅ Saved: topic_consistency_heatmap.png")

plt.close('all')

# ===============================================================================
# SUMMARY
# ===============================================================================

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

print(f"\n   Total binary cases: {len(df_binary)}")
print(f"   Always wrong (0/5): {df_binary['always_wrong'].sum()} ({df_binary['always_wrong'].mean()*100:.1f}%)")
print(f"   Always right (5/5): {df_binary['always_right'].sum()} ({df_binary['always_right'].mean()*100:.1f}%)")

print(f"\n   🏆 HARDEST TOPICS (by always-wrong rate):")
for _, row in topic_stats.head(5).iterrows():
    print(f"      {row['topic']}: {row['always_wrong_rate']*100:.1f}% always wrong (n={row['n_cases']})")

print(f"\n   ✅ EASIEST TOPICS (by always-wrong rate):")
for _, row in topic_stats.tail(5).iterrows():
    print(f"      {row['topic']}: {row['always_wrong_rate']*100:.1f}% always wrong (n={row['n_cases']})")

print(f"\n   📁 All outputs saved to: {OUTPUT_DIR}/")
print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
