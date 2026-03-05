#!/usr/bin/env python3
"""
===============================================================================
MULTI-SEED TOPIC ANALYSIS (FINAL)
===============================================================================

Merges multi-seed consistency results with topic labels to produce
ROBUST topic difficulty rankings based on 5-seed × 5-fold evaluation.

KEY: The consistency.csv has case_id 0-1960 which corresponds to the
FILTERED binary cases, not the original 2466-row pkl. We must filter
the pkl the same way, then merge on case_id.

Inputs:
    - case_consistency.csv (from multi-seed run) - 1961 binary cases
    - sj_231025_w_topics_all_cases.pkl (all 2466 cases with topics)

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

# ===============================================================================
# LOAD DATA
# ===============================================================================

print("="*70)
print("📊 MULTI-SEED TOPIC ANALYSIS")
print("="*70)

# Load consistency data
print(f"\n📂 Loading: {CONSISTENCY_FILE}")
df_consistency = pd.read_csv(CONSISTENCY_FILE)
print(f"   Rows: {len(df_consistency)}")
print(f"   case_id range: {df_consistency['case_id'].min()} to {df_consistency['case_id'].max()}")

# Verify times_correct column exists
if 'times_correct' not in df_consistency.columns:
    print(f"\n❌ ERROR: 'times_correct' column not found!")
    exit(1)

# Load topics data (all cases)
print(f"\n📂 Loading: {TOPICS_FILE}")
df_topics_full = pd.read_pickle(TOPICS_FILE)
print(f"   Rows (all cases): {len(df_topics_full)}")

# ===============================================================================
# FILTER TOPICS TO BINARY CASES (same as multi-seed did)
# ===============================================================================

print("\n" + "="*70)
print("📊 ALIGNING DATASETS")
print("="*70)

print(f"\n   Raw outcome distribution in pkl:")
print(df_topics_full['outcome'].value_counts())

# Normalize outcomes (same logic multi-seed used)
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

df_topics_full['outcome_binary'] = df_topics_full['outcome'].apply(normalize_outcome)

# Filter to binary cases
df_topics = df_topics_full[df_topics_full['outcome_binary'].isin(['GRANTED', 'REFUSED'])].copy()
df_topics = df_topics.reset_index(drop=True)  # Reset index to get 0-1960
df_topics['case_id'] = df_topics.index  # case_id = new sequential index

print(f"\n   Binary cases in pkl: {len(df_topics)}")
print(f"   Binary cases in consistency.csv: {len(df_consistency)}")

# Verify counts match
if len(df_topics) != len(df_consistency):
    print(f"\n⚠️  WARNING: Counts don't match exactly!")
    print(f"   pkl binary: {len(df_topics)}, consistency: {len(df_consistency)}")
    print(f"   Proceeding with merge on case_id...")

# ===============================================================================
# MERGE ON case_id
# ===============================================================================

print(f"\n🔗 Merging on case_id...")
df = df_consistency.merge(df_topics, on='case_id', how='inner')
print(f"   Merged rows: {len(df)}")

if len(df) < len(df_consistency) * 0.95:
    print(f"\n❌ ERROR: Too many rows lost in merge! Check alignment.")
    exit(1)

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

print(f"   ✅ Merged and ready: {len(df)} cases")

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ===============================================================================
# SANITY CHECK COUNTS
# ===============================================================================

print("\n" + "="*70)
print("📊 SANITY CHECK")
print("="*70)

n_always_wrong = df['always_wrong'].sum()
n_always_right = df['always_right'].sum()

print(f"\n   Always-wrong (0/5): {n_always_wrong} cases ({n_always_wrong/len(df)*100:.1f}%)")
print(f"   Always-right (5/5): {n_always_right} cases ({n_always_right/len(df)*100:.1f}%)")
print(f"   GRANTED: {(df['outcome_binary'] == 'GRANTED').sum()}")
print(f"   REFUSED: {(df['outcome_binary'] == 'REFUSED').sum()}")

# ===============================================================================
# CONSISTENCY DISTRIBUTION
# ===============================================================================

print("\n" + "="*70)
print("📊 CONSISTENCY DISTRIBUTION")
print("="*70)

print(f"\n   Distribution of times_correct (0-5):")
print(df[consistency_col].value_counts().sort_index())

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
topic_stats = analyze_topic_errors(df, 'primary_topic', min_cases=20)

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

topic_outcome = analyze_topic_outcome_interaction(df, 'primary_topic')

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
for strata in df['LET_strata'].unique():
    strata_df = df[df['LET_strata'] == strata]
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
for topic in df['primary_topic'].unique():
    if pd.isna(topic):
        continue
    for strata in df['LET_strata'].unique():
        subset = df[(df['primary_topic'] == topic) & (df['LET_strata'] == strata)]
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
        topic_df = df[df['primary_topic'] == topic]
        dist = topic_df[consistency_col].value_counts(normalize=True).sort_index()
        consistency_dist.append({
            'topic': topic,
            **{f'{i}/5': dist.get(i, 0) for i in range(6)}
        })
    
    dist_df = pd.DataFrame(consistency_dist).set_index('topic')
    
    # Use matplotlib for heatmap
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

print(f"\n   Total binary cases: {len(df)}")
print(f"   Always wrong (0/5): {df['always_wrong'].sum()} ({df['always_wrong'].mean()*100:.1f}%)")
print(f"   Always right (5/5): {df['always_right'].sum()} ({df['always_right'].mean()*100:.1f}%)")

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
