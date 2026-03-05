#!/usr/bin/env python3
"""
===============================================================================
MULTI-SEED PHRASE DIFFERENTIAL ANALYSIS (FIXED)
===============================================================================

Identifies linguistic patterns that distinguish structurally hard cases
(always-wrong, 0/5 consistency) from robust cases (always-right, 5/5)
using multi-seed cross-validation results.

Methodology: We contrast cases that were consistently misclassified vs 
consistently correct across 5 seeds (5-fold CV each), enabling identification
of linguistic patterns that are robust to training variance.

Inputs:
    - case_consistency.csv (from multi-seed run) - has 'times_correct' column
    - sj_231025_w_topics_all_cases.pkl (with text and topics)

Outputs:
    - Phrase differential analysis (error-prone vs correct-associated)
    - Topic-controlled consistency analysis
    - Cross-topic error signals vs topic proxies
    - Publication-ready tables with FDR-corrected p-values

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# CONFIGURATION
# ===============================================================================

CONSISTENCY_FILE = "legalbert_multiseed_attention/case_consistency.csv"
TOPICS_FILE = "sj_231025_w_topics_all_cases.pkl"
OUTPUT_DIR = "multiseed_phrase_analysis"

# Analysis parameters
MIN_DOC_FREQ = 5          # Minimum cases containing phrase
MIN_TOPIC_CASES = 10      # Minimum cases per topic for consistency analysis
CONSISTENCY_THRESHOLD = 0.70  # 70% for cross-topic signal
SIGNIFICANCE_LEVEL = 0.05

# Expected counts (sanity check)
EXPECTED_ALWAYS_WRONG = 153
EXPECTED_ALWAYS_RIGHT = 546

# ===============================================================================
# LOAD AND MERGE DATA
# ===============================================================================

print("="*70)
print("📊 MULTI-SEED PHRASE DIFFERENTIAL ANALYSIS")
print("="*70)

# Load consistency data
print(f"\n📂 Loading: {CONSISTENCY_FILE}")
df_consistency = pd.read_csv(CONSISTENCY_FILE)
print(f"   Rows: {len(df_consistency)}")
print(f"   Columns: {list(df_consistency.columns)}")

# Verify times_correct column exists
if 'times_correct' not in df_consistency.columns:
    print(f"\n❌ ERROR: 'times_correct' column not found!")
    print(f"   Available columns: {list(df_consistency.columns)}")
    exit(1)

# Load topics/text data
print(f"\n📂 Loading: {TOPICS_FILE}")
df_topics = pd.read_pickle(TOPICS_FILE)
print(f"   Rows: {len(df_topics)}")

# Check row count alignment
if len(df_consistency) != len(df_topics):
    print(f"\n❌ ERROR: Row counts don't match!")
    print(f"   Consistency: {len(df_consistency)}, Topics: {len(df_topics)}")
    exit(1)

# Verify alignment by checking first few text fields match
print(f"\n🔍 Verifying row alignment...")
n_check = min(100, len(df_topics))
facts_match = sum(
    str(df_topics['facts'].iloc[i])[:100] == str(df_topics['facts'].iloc[i])[:100] 
    for i in range(n_check)
)
print(f"   First {n_check} rows alignment check: {facts_match}/{n_check} match")

if facts_match < n_check * 0.95:
    print(f"\n⚠️  WARNING: Row alignment may be off! Proceeding with caution...")

# Merge on index (verified aligned)
print(f"\n🔗 Merging datasets...")
df = df_consistency.copy()
df['primary_topic'] = df_topics['primary_topic'].values
df['secondary_topic'] = df_topics['secondary_topic'].values
df['outcome'] = df_topics['outcome'].values
df['facts'] = df_topics['facts'].values
df['applicant_reason'] = df_topics['applicant_reason'].values
df['defence_reason'] = df_topics['defence_reason'].values

# Create combined text field for analysis
df['combined_text'] = (
    df['facts'].fillna('') + ' ' + 
    df['applicant_reason'].fillna('') + ' ' + 
    df['defence_reason'].fillna('')
)

# Use times_correct as consistency measure
consistency_col = 'times_correct'
print(f"   Using consistency column: '{consistency_col}'")

# Define categories (0/5 = always wrong, 5/5 = always right)
df['always_wrong'] = df[consistency_col] == 0
df['always_right'] = df[consistency_col] == 5

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

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ===============================================================================
# SANITY CHECK COUNTS
# ===============================================================================

print("\n" + "="*70)
print("📊 SANITY CHECK: ALWAYS-WRONG / ALWAYS-RIGHT COUNTS")
print("="*70)

n_always_wrong = df_binary['always_wrong'].sum()
n_always_right = df_binary['always_right'].sum()

print(f"\n   Always-wrong (0/5): {n_always_wrong} cases")
print(f"   Always-right (5/5): {n_always_right} cases")

if abs(n_always_wrong - EXPECTED_ALWAYS_WRONG) > 5:
    print(f"   ⚠️  Expected ~{EXPECTED_ALWAYS_WRONG} always-wrong, got {n_always_wrong}")
if abs(n_always_right - EXPECTED_ALWAYS_RIGHT) > 10:
    print(f"   ⚠️  Expected ~{EXPECTED_ALWAYS_RIGHT} always-right, got {n_always_right}")

if n_always_wrong < 20 or n_always_right < 20:
    print("\n   ❌ ERROR: Insufficient cases for analysis!")
    exit(1)

# Get always-wrong and always-right cases
wrong_cases = df_binary[df_binary['always_wrong']].copy()
right_cases = df_binary[df_binary['always_right']].copy()

# ===============================================================================
# EXTRACT PHRASES (EFFICIENT: PRECOMPUTE ONCE)
# ===============================================================================

print("\n" + "="*70)
print("📊 STAGE 1: EXTRACTING PHRASES")
print("="*70)

# Extract n-grams using CountVectorizer
vectorizer = CountVectorizer(
    ngram_range=(1, 3),
    min_df=MIN_DOC_FREQ,
    max_df=0.95,
    stop_words='english',
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    lowercase=True
)

# Fit on all binary cases
all_texts = df_binary['combined_text'].tolist()
vectorizer.fit(all_texts)
feature_names = vectorizer.get_feature_names_out()

print(f"   Extracted {len(feature_names)} phrases with doc freq >= {MIN_DOC_FREQ}")

# PRECOMPUTE document-term matrix for ALL cases (efficiency fix)
print(f"   Precomputing document-term matrix...")
X_all = (vectorizer.transform(all_texts) > 0).astype(np.uint8)
print(f"   Matrix shape: {X_all.shape}")

# Create index mapping for efficient slicing
wrong_indices = df_binary[df_binary['always_wrong']].index.tolist()
right_indices = df_binary[df_binary['always_right']].index.tolist()

# Map to positional indices in X_all
idx_to_pos = {idx: pos for pos, idx in enumerate(df_binary.index)}
wrong_pos = [idx_to_pos[idx] for idx in wrong_indices]
right_pos = [idx_to_pos[idx] for idx in right_indices]

# Slice matrices
X_wrong = X_all[wrong_pos, :]
X_right = X_all[right_pos, :]

n_wrong = X_wrong.shape[0]
n_right = X_right.shape[0]

print(f"   Always-wrong matrix: {X_wrong.shape}")
print(f"   Always-right matrix: {X_right.shape}")

# ===============================================================================
# COMPUTE DIFFERENTIAL FREQUENCIES
# ===============================================================================

print("\n" + "="*70)
print("📊 STAGE 2: COMPUTING DIFFERENTIAL FREQUENCIES")
print("="*70)

# Compute frequencies
freq_wrong = np.array(X_wrong.sum(axis=0)).flatten() / n_wrong
freq_right = np.array(X_right.sum(axis=0)).flatten() / n_right

# Compute differential (Δ)
delta = freq_wrong - freq_right

# Compute Fisher's exact test for each phrase
print("\n   Computing Fisher's exact tests...")
p_values = []
odds_ratios = []

for i in range(len(feature_names)):
    # 2x2 contingency table
    a = int(X_wrong[:, i].sum())  # wrong cases with phrase
    b = n_wrong - a               # wrong cases without phrase
    c = int(X_right[:, i].sum())  # right cases with phrase
    d = n_right - c               # right cases without phrase
    
    try:
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
    except:
        odds_ratio, p_value = np.nan, 1.0
    
    p_values.append(p_value)
    odds_ratios.append(odds_ratio)

# Apply Benjamini-Hochberg FDR correction
print("   Applying Benjamini-Hochberg FDR correction...")
p_values = np.array(p_values)
n_tests = len(p_values)
sorted_indices = np.argsort(p_values)
sorted_p = p_values[sorted_indices]

# BH procedure
fdr_values = np.zeros(n_tests)
for i, idx in enumerate(sorted_indices):
    rank = i + 1
    fdr_values[idx] = min(1.0, p_values[idx] * n_tests / rank)

# Ensure monotonicity (cumulative minimum from end)
fdr_sorted = fdr_values[sorted_indices]
for i in range(n_tests - 2, -1, -1):
    fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
fdr_values[sorted_indices] = fdr_sorted

# Create results dataframe
phrase_results = pd.DataFrame({
    'phrase': feature_names,
    'freq_wrong': freq_wrong,
    'freq_right': freq_right,
    'delta': delta,
    'abs_delta': np.abs(delta),
    'p_value': p_values,
    'fdr_q_value': fdr_values,
    'odds_ratio': odds_ratios,
    'significant_raw': p_values < SIGNIFICANCE_LEVEL,
    'significant_fdr': fdr_values < SIGNIFICANCE_LEVEL
})

# Count occurrences
phrase_results['n_wrong'] = np.array(X_wrong.sum(axis=0)).flatten().astype(int)
phrase_results['n_right'] = np.array(X_right.sum(axis=0)).flatten().astype(int)
phrase_results['n_total'] = phrase_results['n_wrong'] + phrase_results['n_right']

n_sig_raw = phrase_results['significant_raw'].sum()
n_sig_fdr = phrase_results['significant_fdr'].sum()
print(f"   Significant (raw p<0.05): {n_sig_raw}")
print(f"   Significant (FDR q<0.05): {n_sig_fdr}")

# ===============================================================================
# TOPIC-CONTROLLED CONSISTENCY ANALYSIS (EFFICIENT)
# ===============================================================================

print("\n" + "="*70)
print("📊 STAGE 3: TOPIC-CONTROLLED CONSISTENCY ANALYSIS")
print("="*70)

# Get topics with sufficient cases
topic_counts = df_binary['primary_topic'].value_counts()
valid_topics = topic_counts[topic_counts >= MIN_TOPIC_CASES].index.tolist()
print(f"\n   Topics with >= {MIN_TOPIC_CASES} cases: {len(valid_topics)}")

# For top phrases, compute within-topic consistency
top_n_phrases = 100
top_phrases_df = phrase_results.nlargest(top_n_phrases, 'abs_delta')
top_phrases = top_phrases_df['phrase'].tolist()
top_phrase_indices = [list(feature_names).index(p) for p in top_phrases]

print(f"   Analyzing top {top_n_phrases} phrases within each topic...")

# Precompute topic masks for efficiency
topic_masks = {}
for topic in valid_topics:
    topic_mask = df_binary['primary_topic'] == topic
    topic_wrong_mask = topic_mask & df_binary['always_wrong']
    topic_right_mask = topic_mask & df_binary['always_right']
    
    # Get positional indices
    topic_wrong_pos = [idx_to_pos[idx] for idx in df_binary[topic_wrong_mask].index]
    topic_right_pos = [idx_to_pos[idx] for idx in df_binary[topic_right_mask].index]
    
    if len(topic_wrong_pos) >= 3 and len(topic_right_pos) >= 3:
        topic_masks[topic] = {
            'wrong_pos': topic_wrong_pos,
            'right_pos': topic_right_pos,
            'n_wrong': len(topic_wrong_pos),
            'n_right': len(topic_right_pos)
        }

print(f"   Topics with sufficient always-wrong AND always-right: {len(topic_masks)}")

consistency_results = []

for phrase, phrase_idx in zip(top_phrases, top_phrase_indices):
    global_delta = phrase_results[phrase_results['phrase'] == phrase]['delta'].values[0]
    global_direction = 'error' if global_delta > 0 else 'correct'
    
    topic_effects = []
    
    for topic, masks in topic_masks.items():
        # Get phrase frequencies within topic
        freq_topic_wrong = X_all[masks['wrong_pos'], phrase_idx].sum() / masks['n_wrong']
        freq_topic_right = X_all[masks['right_pos'], phrase_idx].sum() / masks['n_right']
        
        topic_delta = freq_topic_wrong - freq_topic_right
        
        # Treat delta == 0 as neutral (exclude from direction counting)
        if abs(topic_delta) < 0.001:  # Effectively zero
            direction = 'neutral'
        else:
            direction = 'error' if topic_delta > 0 else 'correct'
        
        topic_effects.append({
            'topic': topic,
            'delta': topic_delta,
            'direction': direction
        })
    
    if len(topic_effects) == 0:
        continue
    
    # Compute consistency (excluding neutral)
    non_neutral = [te for te in topic_effects if te['direction'] != 'neutral']
    if len(non_neutral) == 0:
        consistency = 0.0
    else:
        same_direction = sum(1 for te in non_neutral if te['direction'] == global_direction)
        consistency = same_direction / len(non_neutral)
    
    consistency_results.append({
        'phrase': phrase,
        'global_delta': global_delta,
        'global_direction': global_direction,
        'n_topics_analyzed': len(topic_effects),
        'n_topics_non_neutral': len(non_neutral),
        'n_same_direction': same_direction if len(non_neutral) > 0 else 0,
        'consistency': consistency,
        'is_cross_topic_signal': consistency >= CONSISTENCY_THRESHOLD and global_direction == 'error',
        'is_reliable_signal': consistency >= CONSISTENCY_THRESHOLD and global_direction == 'correct',
        'is_topic_proxy': consistency < CONSISTENCY_THRESHOLD
    })

consistency_df = pd.DataFrame(consistency_results)
print(f"   Computed consistency for {len(consistency_df)} phrases")

# ===============================================================================
# CATEGORIZE PHRASES
# ===============================================================================

print("\n" + "="*70)
print("📊 STAGE 4: CATEGORIZING PHRASES")
print("="*70)

# Merge with main results
phrase_results = phrase_results.merge(
    consistency_df[['phrase', 'consistency', 'n_topics_analyzed', 'n_topics_non_neutral',
                    'is_cross_topic_signal', 'is_reliable_signal', 'is_topic_proxy']],
    on='phrase',
    how='left'
)

# Fill NaN for phrases not in consistency analysis
phrase_results['consistency'] = phrase_results['consistency'].fillna(0)
phrase_results['is_cross_topic_signal'] = phrase_results['is_cross_topic_signal'].fillna(False)
phrase_results['is_reliable_signal'] = phrase_results['is_reliable_signal'].fillna(False)
phrase_results['is_topic_proxy'] = phrase_results['is_topic_proxy'].fillna(True)

# Categorize
def categorize_phrase(row):
    if row['is_cross_topic_signal']:
        return 'CROSS-TOPIC ERROR SIGNAL'
    elif row['is_reliable_signal']:
        return 'RELIABLE SIGNAL (CORRECT)'
    elif row['is_topic_proxy'] and row['delta'] > 0:
        return 'TOPIC PROXY (ERROR)'
    elif row['is_topic_proxy'] and row['delta'] < 0:
        return 'TOPIC PROXY (CORRECT)'
    else:
        return 'UNCATEGORIZED'

phrase_results['category'] = phrase_results.apply(categorize_phrase, axis=1)

# Print summary
category_counts = phrase_results['category'].value_counts()
print(f"\n   Phrase Categories:")
for cat, count in category_counts.items():
    print(f"      {cat}: {count}")

# ===============================================================================
# OUTPUT RESULTS
# ===============================================================================

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)

# Top error-associated phrases
print("\n" + "-"*70)
print("🔴 TOP PHRASES OVERREPRESENTED IN ALWAYS-WRONG CASES (0/5 consistency)")
print("-"*70)

error_phrases = phrase_results[phrase_results['delta'] > 0].nlargest(20, 'delta')
print(f"\n{'Phrase':<30} {'Δ':>7} {'Wrong%':>7} {'Right%':>7} {'p-val':>8} {'FDR-q':>8} {'Cons%':>7} {'Category':<22}")
print("-"*105)

for _, row in error_phrases.iterrows():
    sig = '*' if row['significant_fdr'] else ''
    cons = f"{row['consistency']*100:.0f}%" if pd.notna(row['consistency']) and row['consistency'] > 0 else '—'
    cat = row['category'].replace('CROSS-TOPIC ERROR SIGNAL', '⚠️ CROSS-TOPIC')
    cat = cat.replace('TOPIC PROXY (ERROR)', '📍 Topic Proxy')
    print(f"{row['phrase']:<30} {row['delta']:>+6.3f} {row['freq_wrong']*100:>6.1f}% {row['freq_right']*100:>6.1f}% "
          f"{row['p_value']:>7.4f} {row['fdr_q_value']:>7.4f}{sig} {cons:>7} {cat:<22}")

# Top correct-associated phrases
print("\n" + "-"*70)
print("🟢 TOP PHRASES OVERREPRESENTED IN ALWAYS-RIGHT CASES (5/5 consistency)")
print("-"*70)

correct_phrases = phrase_results[phrase_results['delta'] < 0].nsmallest(20, 'delta')
print(f"\n{'Phrase':<30} {'Δ':>7} {'Wrong%':>7} {'Right%':>7} {'p-val':>8} {'FDR-q':>8} {'Cons%':>7} {'Category':<22}")
print("-"*105)

for _, row in correct_phrases.iterrows():
    sig = '*' if row['significant_fdr'] else ''
    cons = f"{row['consistency']*100:.0f}%" if pd.notna(row['consistency']) and row['consistency'] > 0 else '—'
    cat = row['category'].replace('RELIABLE SIGNAL (CORRECT)', '✅ RELIABLE')
    cat = cat.replace('TOPIC PROXY (CORRECT)', '📍 Topic Proxy')
    print(f"{row['phrase']:<30} {row['delta']:>+6.3f} {row['freq_wrong']*100:>6.1f}% {row['freq_right']*100:>6.1f}% "
          f"{row['p_value']:>7.4f} {row['fdr_q_value']:>7.4f}{sig} {cons:>7} {cat:<22}")

# Cross-topic error signals
print("\n" + "-"*70)
print("⚠️  CROSS-TOPIC ERROR SIGNALS (Consistency >= 70%, Δ > 0)")
print("-"*70)

cross_topic_errors = phrase_results[phrase_results['is_cross_topic_signal'] == True].sort_values('delta', ascending=False)
if len(cross_topic_errors) > 0:
    for _, row in cross_topic_errors.iterrows():
        sig_marker = " (FDR sig)" if row['significant_fdr'] else ""
        print(f"   '{row['phrase']}': Δ={row['delta']:+.3f}, consistency={row['consistency']*100:.0f}%{sig_marker}")
else:
    print("   (None found - all error-associated phrases are topic-specific)")

# Reliable signals
print("\n" + "-"*70)
print("✅ RELIABLE SIGNALS (Consistency >= 70%, Δ < 0)")
print("-"*70)

reliable_signals = phrase_results[phrase_results['is_reliable_signal'] == True].sort_values('delta')
if len(reliable_signals) > 0:
    for _, row in reliable_signals.head(10).iterrows():
        sig_marker = " (FDR sig)" if row['significant_fdr'] else ""
        print(f"   '{row['phrase']}': Δ={row['delta']:+.3f}, consistency={row['consistency']*100:.0f}%{sig_marker}")
else:
    print("   (None found)")

# ===============================================================================
# SAVE OUTPUTS
# ===============================================================================

print("\n" + "="*70)
print("📁 SAVING OUTPUTS")
print("="*70)

# Full results
phrase_results.to_csv(f"{OUTPUT_DIR}/phrase_analysis_full.csv", index=False)
print(f"   ✅ phrase_analysis_full.csv ({len(phrase_results)} phrases)")

# Top error phrases
error_phrases.to_csv(f"{OUTPUT_DIR}/top_error_phrases.csv", index=False)
print(f"   ✅ top_error_phrases.csv")

# Top correct phrases
correct_phrases.to_csv(f"{OUTPUT_DIR}/top_correct_phrases.csv", index=False)
print(f"   ✅ top_correct_phrases.csv")

# Summary table (for paper)
summary_data = []

# Cross-topic error signals
cts = phrase_results[phrase_results['is_cross_topic_signal'] == True]
if len(cts) > 0:
    summary_data.append({
        'Category': 'Cross-Topic Error Signal',
        'N_Phrases': len(cts),
        'Example_Phrases': ', '.join(cts.nlargest(3, 'delta')['phrase'].tolist()),
        'Delta_Range': f"+{cts['delta'].min():.3f} to +{cts['delta'].max():.3f}",
        'Consistency': f"{cts['consistency'].min()*100:.0f}–{cts['consistency'].max()*100:.0f}%",
        'N_FDR_Significant': cts['significant_fdr'].sum()
    })

# Topic proxy (error)
tpe = phrase_results[(phrase_results['is_topic_proxy'] == True) & (phrase_results['delta'] > 0)]
if len(tpe) > 0:
    top_tpe = tpe.nlargest(5, 'delta')
    summary_data.append({
        'Category': 'Topic Proxy (Error)',
        'N_Phrases': len(tpe),
        'Example_Phrases': ', '.join(top_tpe['phrase'].tolist()),
        'Delta_Range': f"+{tpe['delta'].min():.3f} to +{tpe['delta'].max():.3f}",
        'Consistency': f"{tpe['consistency'].min()*100:.0f}–{tpe['consistency'].max()*100:.0f}%",
        'N_FDR_Significant': tpe['significant_fdr'].sum()
    })

# Reliable signals
rs = phrase_results[phrase_results['is_reliable_signal'] == True]
if len(rs) > 0:
    summary_data.append({
        'Category': 'Reliable Signal (Correct)',
        'N_Phrases': len(rs),
        'Example_Phrases': ', '.join(rs.nsmallest(3, 'delta')['phrase'].tolist()),
        'Delta_Range': f"{rs['delta'].max():.3f} to {rs['delta'].min():.3f}",
        'Consistency': f"{rs['consistency'].min()*100:.0f}–{rs['consistency'].max()*100:.0f}%",
        'N_FDR_Significant': rs['significant_fdr'].sum()
    })

# Topic proxy (correct)
tpc = phrase_results[(phrase_results['is_topic_proxy'] == True) & (phrase_results['delta'] < 0)]
if len(tpc) > 0:
    top_tpc = tpc.nsmallest(5, 'delta')
    summary_data.append({
        'Category': 'Topic Proxy (Correct)',
        'N_Phrases': len(tpc),
        'Example_Phrases': ', '.join(top_tpc['phrase'].tolist()),
        'Delta_Range': f"{tpc['delta'].max():.3f} to {tpc['delta'].min():.3f}",
        'Consistency': f"{tpc['consistency'].min()*100:.0f}–{tpc['consistency'].max()*100:.0f}%",
        'N_FDR_Significant': tpc['significant_fdr'].sum()
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{OUTPUT_DIR}/phrase_summary_table.csv", index=False)
print(f"   ✅ phrase_summary_table.csv (Table 4.1 format)")

# ===============================================================================
# VISUALIZATIONS
# ===============================================================================

print("\n" + "="*70)
print("📊 GENERATING PLOTS")
print("="*70)

# Plot 1: Top error vs correct phrases
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Error phrases
ax1 = axes[0]
top_err = phrase_results.nlargest(15, 'delta')
colors = ['darkred' if row['is_cross_topic_signal'] else 'salmon' for _, row in top_err.iterrows()]
bars = ax1.barh(range(len(top_err)), top_err['delta'] * 100, color=colors)
ax1.set_yticks(range(len(top_err)))
ax1.set_yticklabels(top_err['phrase'])
ax1.set_xlabel('Δ (percentage points)')
ax1.set_title('Phrases Overrepresented in ALWAYS-WRONG Cases\n(Dark Red = Cross-Topic Signal, FDR-corrected)')
ax1.invert_yaxis()
ax1.axvline(x=0, color='black', linewidth=0.5)

# Correct phrases
ax2 = axes[1]
top_corr = phrase_results.nsmallest(15, 'delta')
colors = ['darkgreen' if row['is_reliable_signal'] else 'lightgreen' for _, row in top_corr.iterrows()]
bars = ax2.barh(range(len(top_corr)), top_corr['delta'] * 100, color=colors)
ax2.set_yticks(range(len(top_corr)))
ax2.set_yticklabels(top_corr['phrase'])
ax2.set_xlabel('Δ (percentage points)')
ax2.set_title('Phrases Overrepresented in ALWAYS-RIGHT Cases\n(Dark Green = Reliable Signal, FDR-corrected)')
ax2.invert_yaxis()
ax2.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/phrase_differential_plot.png", dpi=150, bbox_inches='tight')
print(f"   ✅ phrase_differential_plot.png")

# Plot 2: Delta vs Consistency scatter
fig, ax = plt.subplots(figsize=(12, 8))

# Only plot phrases with consistency data
plot_df = phrase_results[phrase_results['consistency'] > 0].copy()

# Color by category
colors = []
for _, row in plot_df.iterrows():
    if row['is_cross_topic_signal']:
        colors.append('darkred')
    elif row['is_reliable_signal']:
        colors.append('darkgreen')
    elif row['delta'] > 0:
        colors.append('lightsalmon')
    else:
        colors.append('lightgreen')

ax.scatter(plot_df['delta'] * 100, plot_df['consistency'] * 100, 
           c=colors, alpha=0.6, s=50)

# Add threshold lines
ax.axhline(y=70, color='gray', linestyle='--', linewidth=1, label='70% consistency threshold')
ax.axvline(x=0, color='black', linewidth=0.5)

# Label significant cross-topic signals
for _, row in plot_df[plot_df['is_cross_topic_signal'] == True].iterrows():
    ax.annotate(row['phrase'], (row['delta']*100, row['consistency']*100),
                fontsize=9, ha='left')

# Label reliable signals
for _, row in plot_df[plot_df['is_reliable_signal'] == True].head(5).iterrows():
    ax.annotate(row['phrase'], (row['delta']*100, row['consistency']*100),
                fontsize=9, ha='right')

ax.set_xlabel('Δ (percentage points): positive = error-prone, negative = correct-associated')
ax.set_ylabel('Consistency across topics (%)')
ax.set_title('Phrase Analysis: Differential Frequency vs Cross-Topic Consistency\n(Multi-Seed: cases consistently misclassified/correct across 5 seeds)')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/delta_vs_consistency.png", dpi=150, bbox_inches='tight')
print(f"   ✅ delta_vs_consistency.png")

plt.close('all')

# ===============================================================================
# FINAL SUMMARY
# ===============================================================================

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

print(f"\n   Analysis based on multi-seed consistency:")
print(f"   - Always-wrong (0/5 seeds correct): {n_always_wrong} cases")
print(f"   - Always-right (5/5 seeds correct): {n_always_right} cases")
print(f"   - Phrases analyzed: {len(phrase_results)}")
print(f"   - FDR-significant phrases: {phrase_results['significant_fdr'].sum()}")

print(f"\n   🔴 CROSS-TOPIC ERROR SIGNALS: {len(cross_topic_errors)}")
if len(cross_topic_errors) > 0:
    for _, row in cross_topic_errors.head(5).iterrows():
        print(f"      '{row['phrase']}': Δ=+{row['delta']:.3f}, consistency={row['consistency']*100:.0f}%")

print(f"\n   🟢 RELIABLE SIGNALS: {len(reliable_signals)}")
if len(reliable_signals) > 0:
    for _, row in reliable_signals.head(5).iterrows():
        print(f"      '{row['phrase']}': Δ={row['delta']:.3f}, consistency={row['consistency']*100:.0f}%")

print(f"\n   📍 TOPIC PROXIES (apparent signals that don't generalize): {len(tpe) + len(tpc)}")

print(f"\n   📁 All outputs saved to: {OUTPUT_DIR}/")

print("\n" + "="*70)
print("✅ PHRASE ANALYSIS COMPLETE")
print("="*70)
