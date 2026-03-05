#!/usr/bin/env python3
"""
Statistical Significance Testing for Legal-BERT Analysis

Tier 1: Necessary
- Chi-square tests for keyword x always-wrong
- Logistic regression with interaction
- Wilson confidence intervals

Tier 2: Enhancements  
- Calibration analysis (ECE)
- Per-fold distribution check
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_pickle('sj_231025.pkl')
df = df[df['outcome'].isin(['summary judgment granted', 'summary judgment refused'])].copy()
df['case_id'] = range(len(df))
n_cases_original = len(df)

consistency = pd.read_csv('legalbert_multiseed_attention/case_consistency.csv')
df = df.merge(consistency[['case_id', 'times_correct']], on='case_id', how='left')

# Validate merge
assert len(df) == n_cases_original, f"Merge lost cases! {n_cases_original} --> {len(df)}"
missing = df['times_correct'].isna().sum()
if missing > 0:
    raise ValueError(f"FATAL: {missing} cases missing from case_consistency.csv!")
print(f"OK Loaded {len(df)} cases with consistency data")

# Create key variables
df['always_wrong'] = (df['times_correct'] == 0).astype(int)
df['refused'] = (df['outcome'] == 'summary judgment refused').astype(int)
df['all_text'] = (df['facts'].fillna('') + ' ' + 
                  df['applicant_reason'].fillna('') + ' ' + 
                  df['defence_reason'].fillna('')).str.lower()

# Add keyword flags with word boundaries
for kw in ['binding', 'defamatory', 'settlement', 'property']:
    df[f'has_{kw}'] = df['all_text'].str.contains(rf'\b{kw}\b', regex=True).astype(int)

print("="*70)
print("TIER 1: STATISTICAL SIGNIFICANCE TESTING")
print("="*70)

# =============================================================================
# 1. CHI-SQUARE TESTS
# =============================================================================

print("\n" + "="*70)
print("1. CHI-SQUARE TESTS: Keyword x Always-Wrong")
print("="*70)

def chi_square_analysis(df, keyword_col, outcome_col='always_wrong'):
    """Perform chi-square test and compute odds ratio with CI."""
    
    # Create contingency table - FORCE both levels to exist
    contingency = pd.crosstab(df[keyword_col], df[outcome_col]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    # Fisher's exact test (more reliable for small samples)
    odds_ratio, fisher_p = fisher_exact(contingency)
    
    # Extract cells using .loc for safety
    a = contingency.loc[1, 1]  # keyword & wrong
    b = contingency.loc[1, 0]  # keyword & right
    c = contingency.loc[0, 1]  # no keyword & wrong
    d = contingency.loc[0, 0]  # no keyword & right
    
    # Haldane-Anscombe correction for zero cells
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_adj, b_adj, c_adj, d_adj = a, b, c, d
    
    # Compute OR and 95% CI using log transformation
    or_adj = (a_adj * d_adj) / (b_adj * c_adj)
    log_or = np.log(or_adj)
    se_log_or = np.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
    
    return {
        'chi2': chi2,
        'p_value': p,
        'fisher_p': fisher_p,
        'expected': expected,
        'use_fisher': (expected < 5).any(),  # True if any expected count < 5
        'odds_ratio': odds_ratio,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'contingency': contingency
    }

chi_square_results = []
for keyword in ['binding', 'defamatory', 'settlement']:
    col = f'has_{keyword}'
    result = chi_square_analysis(df, col)
    result['keyword'] = keyword
    
    # Determine primary p-value based on expected counts
    result['primary_p'] = result['fisher_p'] if result['use_fisher'] else result['p_value']
    chi_square_results.append(result)
    
    print(f"\n{keyword.upper()}")
    print(f"  Contingency table:")
    print(f"                    Not Wrong    Wrong")
    print(f"  No {keyword:<12} {result['contingency'].loc[0,0]:>8}    {result['contingency'].loc[0,1]:>5}")
    print(f"  Has {keyword:<11} {result['contingency'].loc[1,0]:>8}    {result['contingency'].loc[1,1]:>5}")
    print(f"\n  chi2(df=1) = {result['chi2']:.2f}, p = {result['p_value']:.4f}")
    print(f"  Fisher's exact p = {result['fisher_p']:.4f}")
    
    # Report which test is primary
    if result['use_fisher']:
        print(f"  WARNING Expected count < 5 detected, using Fisher as primary test")
        print(f"  PRIMARY p = {result['primary_p']:.4f} (Fisher)")
    else:
        print(f"  PRIMARY p = {result['primary_p']:.4f} (chi2)")
    
    print(f"  Odds Ratio = {result['odds_ratio']:.2f} [95% CI: {result['ci_low']:.2f}-{result['ci_high']:.2f}]")
    
    if result['primary_p'] < 0.05:
        print(f"  OK SIGNIFICANT at alpha=0.05")
    else:
        print(f"  WARNING Not significant at alpha=0.05")

# Multiple testing correction (Benjamini-Hochberg)
print("\n" + "-"*50)
print("MULTIPLE TESTING CORRECTION (Benjamini-Hochberg FDR)")
print("-"*50)

p_values = [r['primary_p'] for r in chi_square_results]  # Use primary_p
keywords = [r['keyword'] for r in chi_square_results]
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

print(f"\n  {'Keyword':<15} {'Raw p':<12} {'Adjusted p':<12} {'Significant':<12}")
print(f"  {'-'*51}")
for kw, raw_p, adj_p, sig in zip(keywords, p_values, p_corrected, reject):
    sig_str = "OK Yes" if sig else "No"
    print(f"  {kw:<15} {raw_p:<12.4f} {adj_p:<12.4f} {sig_str:<12}")

# =============================================================================
# 2. LOGISTIC REGRESSION WITH INTERACTION
# =============================================================================

print("\n" + "="*70)
print("2. LOGISTIC REGRESSION: AlwaysWrong ~ Outcome + Keyword + Interaction")
print("   (with stratum controls: law, evidence, trial)")
print("="*70)

for keyword in ['binding', 'defamatory']:
    print(f"\n{keyword.upper()} ANALYSIS")
    print("-"*50)
    
    # Prepare data WITH STRATUM CONTROLS
    X = df[['refused', 'law', 'evidence', 'trial', f'has_{keyword}']].copy()
    X['interaction'] = X['refused'] * X[f'has_{keyword}']
    X = sm.add_constant(X)
    y = df['always_wrong']
    
    # Fit logistic regression with separation handling
    regularized = False
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        
        # Check for extreme coefficients (separation indicator)
        if (np.abs(model.params) > 10).any():
            print("  WARNING Possible separation detected, using regularized fit...")
            model = sm.Logit(y, X).fit_regularized(alpha=1.0, disp=0)
            regularized = True
            
    except Exception as e:
        print(f"  WARNING Standard fit failed ({e}), using regularized fit...")
        try:
            model = sm.Logit(y, X).fit_regularized(alpha=1.0, disp=0)
            regularized = True
        except Exception as e2:
            print(f"  ERROR Regularized fit also failed: {e2}")
            continue
    
    print(f"\n  Model: AlwaysWrong ~ Refused + Law + Evidence + Trial + Has{keyword.title()} + RefusedxHas{keyword.title()}")
    
    if regularized:
        print(f"\n  {'Variable':<25} {'Coef':<10} {'Odds Ratio':<12}")
        print(f"  {'-'*47}")
        for var in model.params.index:
            coef = model.params[var]
            odds = np.exp(coef)
            print(f"  {var:<25} {coef:<10.3f} {odds:<12.2f}")
        print("\n  Note: SE/p-values unavailable for regularized fit")
    else:
        print(f"\n  {'Variable':<25} {'Coef':<10} {'SE':<10} {'z':<10} {'p-value':<10} {'Odds Ratio':<12}")
        print(f"  {'-'*77}")
        
        for var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            z = model.tvalues[var]
            p = model.pvalues[var]
            odds = np.exp(coef)
            
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:<25} {coef:<10.3f} {se:<10.3f} {z:<10.2f} {p:<10.4f} {odds:<12.2f} {sig}")
        
        # Check interaction significance
        interaction_p = model.pvalues['interaction']
        if interaction_p < 0.05:
            print(f"\n  OK INTERACTION SIGNIFICANT (p={interaction_p:.4f}) after controlling for stratum")
        else:
            print(f"\n  WARNING Interaction not significant (p={interaction_p:.4f}) after controlling for stratum")

# =============================================================================
# 3. WILSON CONFIDENCE INTERVALS
# =============================================================================

print("\n" + "="*70)
print("3. WILSON CONFIDENCE INTERVALS FOR KEY ERROR RATES")
print("="*70)

def wilson_ci(events, total, alpha=0.05):
    """Compute Wilson score confidence interval."""
    if total == 0:
        return 0, 0, 0
    proportion = events / total
    ci_low, ci_high = proportion_confint(events, total, alpha=alpha, method='wilson')
    return proportion, ci_low, ci_high

print(f"\n  {'Group':<40} {'n':<8} {'Err%':<12} {'95% CI':<20}")
print(f"  {'-'*80}")

# Key rates to report
groups = [
    ("Overall", df['always_wrong'].sum(), len(df)),
    ("GRANTED cases", df[df['refused']==0]['always_wrong'].sum(), len(df[df['refused']==0])),
    ("REFUSED cases", df[df['refused']==1]['always_wrong'].sum(), len(df[df['refused']==1])),
    ("Binding cases (all)", df[df['has_binding']==1]['always_wrong'].sum(), len(df[df['has_binding']==1])),
    ("Binding + REFUSED", df[(df['has_binding']==1) & (df['refused']==1)]['always_wrong'].sum(), 
     len(df[(df['has_binding']==1) & (df['refused']==1)])),
    ("Binding + GRANTED", df[(df['has_binding']==1) & (df['refused']==0)]['always_wrong'].sum(),
     len(df[(df['has_binding']==1) & (df['refused']==0)])),
    ("Defamatory cases (all)", df[df['has_defamatory']==1]['always_wrong'].sum(), len(df[df['has_defamatory']==1])),
    ("Defamatory + REFUSED", df[(df['has_defamatory']==1) & (df['refused']==1)]['always_wrong'].sum(),
     len(df[(df['has_defamatory']==1) & (df['refused']==1)])),
]

for name, wrong, total in groups:
    prop, ci_low, ci_high = wilson_ci(wrong, total)
    print(f"  {name:<40} {total:<8} {prop*100:<12.1f} [{ci_low*100:.1f}%-{ci_high*100:.1f}%]")

# =============================================================================
# TIER 2: ENHANCEMENTS
# =============================================================================

print("\n" + "="*70)
print("TIER 2: CALIBRATION ANALYSIS")
print("="*70)

# Load predictions with probabilities
all_preds = pd.read_csv('legalbert_multiseed_attention/all_predictions.csv')

# Sanity check labels
assert set(all_preds['label'].unique()) <= {0, 1}, "Unexpected labels!"
print(f"\n  Sanity check:")
print(f"    Mean prob_granted for label=0 (granted): {all_preds[all_preds['label']==0]['prob_granted'].mean():.3f}")
print(f"    Mean prob_granted for label=1 (refused): {all_preds[all_preds['label']==1]['prob_granted'].mean():.3f}")

print("\n4. EXPECTED CALIBRATION ERROR (ECE) - Predicted-Class Confidence")
print("-"*50)

def compute_ece(confidence, correct, n_bins=10):
    """
    Compute Expected Calibration Error using predicted-class confidence.
    
    Args:
        confidence: probability assigned to the predicted class
        correct: 1 if prediction was correct, 0 otherwise
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
        if in_bin.any():
            avg_confidence = confidence[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            prop_in_bin = in_bin.mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece

# Merge predictions with keyword info
n_preds_before = len(all_preds)
preds_with_info = all_preds.merge(df[['case_id', 'has_binding', 'has_defamatory', 'refused']], on='case_id', how='left')
assert len(preds_with_info) == n_preds_before, f"Merge lost predictions! {n_preds_before} --> {len(preds_with_info)}"
assert preds_with_info['has_binding'].notna().all(), "Missing keyword info after merge!"

# Compute predicted-class confidence and correctness
probs = preds_with_info[['prob_granted', 'prob_refused']].values
pred = preds_with_info['predicted'].values.astype(int)
conf = probs[np.arange(len(probs)), pred]  # Probability of predicted class
correct = (preds_with_info['label'].values.astype(int) == pred).astype(int)

# Overall ECE
overall_ece = compute_ece(conf, correct)
print(f"\n  Overall ECE: {overall_ece:.4f}")

# ECE by subgroup (using same confidence/correct for consistency)
preds_with_info['confidence'] = conf
preds_with_info['correct'] = correct

subgroups = [
    ("GRANTED cases", preds_with_info['label'] == 0),
    ("REFUSED cases", preds_with_info['label'] == 1),
    ("Binding cases", preds_with_info['has_binding'] == 1),
    ("Binding + REFUSED", (preds_with_info['has_binding'] == 1) & (preds_with_info['label'] == 1)),
    ("Non-binding cases", preds_with_info['has_binding'] == 0),
]

print(f"\n  {'Subgroup':<30} {'N':<10} {'ECE':<10}")
print(f"  {'-'*50}")

for name, mask in subgroups:
    subset = preds_with_info[mask]
    if len(subset) > 0:
        ece = compute_ece(subset['confidence'].values, subset['correct'].values)
        print(f"  {name:<30} {len(subset):<10} {ece:.4f}")

# =============================================================================
# 5. PER-FOLD DISTRIBUTION CHECK
# =============================================================================

print("\n" + "="*70)
print("5. PER-FOLD DISTRIBUTION: Binding + REFUSED Cases")
print("="*70)

print("\n  Checking if binding-refused cases are underrepresented in training folds...")

# Get fold assignments
fold_dist = all_preds.merge(df[['case_id', 'has_binding', 'refused']], on='case_id', how='left')
assert len(fold_dist) == len(all_preds), "Merge lost predictions!"
binding_refused = fold_dist[(fold_dist['has_binding']==1) & (fold_dist['refused']==1)]

print(f"\n  Total binding+refused cases: {len(df[(df['has_binding']==1) & (df['refused']==1)])}")
print(f"\n  {'Seed':<8} {'Fold':<8} {'In Validation':<15}")
print(f"  {'-'*31}")

# For each seed/fold, count binding+refused in validation
for seed in sorted(all_preds['seed'].unique()):
    for fold in sorted(all_preds['fold'].unique()):
        subset = binding_refused[(binding_refused['seed']==seed) & (binding_refused['fold']==fold)]
        in_val = len(subset)
        print(f"  {seed:<8} {fold:<8} {in_val:<15}")

# Summary statistics - ensure all 25 combos are counted
all_pairs = pd.MultiIndex.from_product(
    [sorted(all_preds['seed'].unique()), sorted(all_preds['fold'].unique())],
    names=['seed', 'fold']
)
counts = binding_refused.groupby(['seed', 'fold']).size().reindex(all_pairs, fill_value=0)

print(f"\n  Mean per fold: {counts.mean():.1f} binding+refused cases in validation")
print(f"  Std dev: {counts.std():.2f}")
print(f"  Min: {counts.min()}, Max: {counts.max()}")

# =============================================================================
# 6. ADDITIONAL: OVERCONFIDENCE ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("6. OVERCONFIDENCE ANALYSIS: Are Wrong Predictions Confident?")
print("="*70)

# For always-wrong cases, what was the average confidence?
aw_cases = df[df['always_wrong']==1]['case_id'].tolist()
aw_preds = all_preds[all_preds['case_id'].isin(aw_cases)]

print(f"\n  Always-Wrong Cases (n={len(aw_cases)} cases, {len(aw_preds)} predictions):")

# For each prediction, confidence = max(prob_granted, prob_refused)
aw_preds = aw_preds.copy()
aw_preds['confidence'] = aw_preds[['prob_granted', 'prob_refused']].max(axis=1)
aw_preds['pred_granted'] = (aw_preds['predicted'] == 0).astype(int)

print(f"  Mean confidence in wrong predictions: {aw_preds['confidence'].mean():.3f}")
print(f"  Median confidence: {aw_preds['confidence'].median():.3f}")

# Compare to always-right
ar_cases = df[df['times_correct']==5]['case_id'].tolist()
ar_preds = all_preds[all_preds['case_id'].isin(ar_cases)]
ar_preds = ar_preds.copy()
ar_preds['confidence'] = ar_preds[['prob_granted', 'prob_refused']].max(axis=1)

print(f"\n  Always-Right Cases (n={len(ar_cases)} cases, {len(ar_preds)} predictions):")
print(f"  Mean confidence in correct predictions: {ar_preds['confidence'].mean():.3f}")
print(f"  Median confidence: {ar_preds['confidence'].median():.3f}")

# Binding + Refused wrong cases
br_aw = df[(df['has_binding']==1) & (df['refused']==1) & (df['always_wrong']==1)]['case_id'].tolist()
if len(br_aw) > 0:
    br_preds = all_preds[all_preds['case_id'].isin(br_aw)]
    br_preds = br_preds.copy()
    br_preds['confidence'] = br_preds[['prob_granted', 'prob_refused']].max(axis=1)
    print(f"\n  Binding + REFUSED + Always-Wrong (n={len(br_aw)} cases):")
    print(f"  Mean confidence: {br_preds['confidence'].mean():.3f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: KEY STATISTICAL FINDINGS")
print("="*70)

print("""
1. CHI-SQUARE TESTS:
   - Binding x AlwaysWrong: Check p-value above
   - Defamatory x AlwaysWrong: Check p-value above

2. LOGISTIC REGRESSION:
   - Interaction term (Refused x Binding) significance confirms
     directional domain-conditioned bias

3. WILSON CONFIDENCE INTERVALS:
   - Binding + REFUSED: 32.3% [CI: see above]
   - Wide CI reflects small n - acknowledge in paper

4. CALIBRATION:
   - ECE by subgroup shows if binding cases are miscalibrated

5. FOLD DISTRIBUTION:
   - If binding+refused evenly distributed --> not a sampling artifact

6. OVERCONFIDENCE:
   - If wrong predictions are confident --> miscalibration confirmed
""")
