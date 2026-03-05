#!/usr/bin/env python3
"""
===============================================================================
FIXED ATTENTION EXTRACTION (v2)
===============================================================================

Properly extracts attention using CORRECT methodology:

FIXES APPLIED:
1. Uses the CORRECT MODEL per case (matching seed/fold from all_predictions.csv)
2. Reports RAW DENSITY (not normalized to sum=1)
3. Reports BOTH mean and max across heads
4. Uses MULTIPLE [MASK] tokens for ablation
5. Verifies label mapping from model.config
6. Special tokens already added during training (verified)

Usage:
    python fix_attention_v2.py
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===============================================================================
# CONFIGURATION
# ===============================================================================

DATA_DIR = Path("legalbert_multiseed_attention")
ORIGINAL_DATA = "sj_231025.pkl"
OUTPUT_DIR = Path("attention_fixed_v2")
MAX_LENGTH = 512
MASK_LENGTH = 50  # Number of [MASK] tokens to use

# ===============================================================================
# GPU CHECK
# ===============================================================================

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ No GPU available. This will be slow.")
        return torch.device('cpu')
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    return torch.device('cuda')

# ===============================================================================
# TEXT FORMATTING
# ===============================================================================

def format_text(row):
    facts = str(row.get('facts', '') or '').strip()
    applicant = str(row.get('applicant_reason', '') or '').strip()
    defence = str(row.get('defence_reason', '') or '').strip()
    return f"[FACTS] {facts} [APPLICANT] {applicant} [DEFENCE] {defence}"


def mask_section(text, section, mask_token="[MASK]", n_masks=MASK_LENGTH):
    """Replace a section's content with multiple [MASK] tokens."""
    mask_str = " ".join([mask_token] * n_masks)
    
    if section == 'FACTS':
        return re.sub(r'\[FACTS\].*?\[APPLICANT\]', f'[FACTS] {mask_str} [APPLICANT]', text, flags=re.DOTALL)
    elif section == 'APPLICANT':
        return re.sub(r'\[APPLICANT\].*?\[DEFENCE\]', f'[APPLICANT] {mask_str} [DEFENCE]', text, flags=re.DOTALL)
    elif section == 'DEFENCE':
        return re.sub(r'\[DEFENCE\].*$', f'[DEFENCE] {mask_str}', text, flags=re.DOTALL)
    return text

# ===============================================================================
# ATTENTION EXTRACTION (FIXED)
# ===============================================================================

def extract_attention(model, tokenizer, text, device):
    """
    Extract CLS attention from last layer.
    Returns BOTH mean and max across heads.
    """
    model.eval()
    
    encoding = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                         padding='max_length', return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    
    # Safety check: ensure attentions were returned
    if outputs.attentions is None or len(outputs.attentions) == 0:
        raise RuntimeError(
            "No attentions returned. Ensure model loaded with attn_implementation='eager'."
        )
    
    # Last layer: (batch, heads, seq_len, seq_len)
    attention = outputs.attentions[-1]
    
    # CLS attention (row 0) per head: (heads, seq_len)
    cls_attn_per_head = attention[0, :, 0, :].cpu().numpy()
    
    # Mean AND max across heads
    cls_attn_mean = cls_attn_per_head.mean(axis=0)
    cls_attn_max = cls_attn_per_head.max(axis=0)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    seq_len = attention_mask.sum().item()
    
    return {
        'tokens': tokens[:seq_len],
        'cls_attention_mean': cls_attn_mean[:seq_len],
        'cls_attention_max': cls_attn_max[:seq_len],
        'cls_attention_per_head': cls_attn_per_head[:, :seq_len],
    }


def compute_section_metrics(tokens, attention_mean, attention_max):
    """
    Compute RAW DENSITY (mean attention per token) - NOT normalized.
    Also computes max-based metrics.
    """
    section_sum_mean = {'FACTS': 0.0, 'APPLICANT': 0.0, 'DEFENCE': 0.0, 'OTHER': 0.0}
    section_sum_max = {'FACTS': 0.0, 'APPLICANT': 0.0, 'DEFENCE': 0.0, 'OTHER': 0.0}
    section_count = {'FACTS': 0, 'APPLICANT': 0, 'DEFENCE': 0, 'OTHER': 0}
    current_section = 'OTHER'
    
    for token, attn_mean, attn_max in zip(tokens, attention_mean, attention_max):
        if token == '[FACTS]':
            current_section = 'FACTS'
        elif token == '[APPLICANT]':
            current_section = 'APPLICANT'
        elif token == '[DEFENCE]':
            current_section = 'DEFENCE'
        elif token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        else:
            section_sum_mean[current_section] += attn_mean
            section_sum_max[current_section] += attn_max
            section_count[current_section] += 1
    
    # RAW DENSITY: mean attention per token (NOT normalized to sum=1)
    raw_density_mean = {}
    raw_density_max = {}
    for section in ['FACTS', 'APPLICANT', 'DEFENCE', 'OTHER']:
        if section_count[section] > 0:
            raw_density_mean[section] = section_sum_mean[section] / section_count[section]
            raw_density_max[section] = section_sum_max[section] / section_count[section]
        else:
            raw_density_mean[section] = 0.0
            raw_density_max[section] = 0.0
    
    # Also compute proportion (for comparison/artifact verification)
    total_sum = sum(section_sum_mean.values())
    if total_sum > 0:
        proportion = {k: v / total_sum for k, v in section_sum_mean.items()}
    else:
        proportion = {k: 0.0 for k in section_sum_mean}
    
    return {
        'raw_density_mean': raw_density_mean,  # THE MEANINGFUL METRIC
        'raw_density_max': raw_density_max,     # MAX-based density
        'proportion': proportion,               # For artifact verification
        'token_counts': section_count,
    }


def get_label_indices(model):
    """Extract correct label indices by searching id2label for substrings."""
    id2label = getattr(model.config, 'id2label', {0: 'LABEL_0', 1: 'LABEL_1'})
    
    granted_idx = 0  # default
    refused_idx = 1  # default
    
    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if 'grant' in label_lower:
            granted_idx = int(idx)
        elif 'refus' in label_lower or 'denied' in label_lower:
            refused_idx = int(idx)
    
    return granted_idx, refused_idx


def verify_special_tokens(tokenizer):
    """Verify special tokens are in vocabulary (not UNK)."""
    print(f"\n   Verifying special tokens:")
    unk_id = tokenizer.unk_token_id
    all_ok = True
    
    for token in ["[FACTS]", "[APPLICANT]", "[DEFENCE]"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        status = "✅" if token_id != unk_id else "❌ (UNK!)"
        print(f"      {token} -> ID {token_id} {status}")
        if token_id == unk_id:
            all_ok = False
    
    if not all_ok:
        print("   ⚠️ WARNING: Some special tokens are UNK! Section detection will fail.")
    
    return all_ok


def get_prediction_probs(model, tokenizer, text, device, granted_idx, refused_idx):
    """Get prediction probabilities with verified label mapping."""
    model.eval()
    
    encoding = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                         padding='max_length', return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    pred_idx = torch.argmax(outputs.logits, dim=-1).item()
    
    return {
        'pred': pred_idx,
        'prob_granted': probs[0, granted_idx].item(),
        'prob_refused': probs[0, refused_idx].item(),
    }


# ===============================================================================
# MODEL CACHE
# ===============================================================================

class ModelCache:
    """Cache models to avoid reloading. Keeps max N models in memory."""
    
    def __init__(self, device, max_models=5):
        self.cache = {}
        self.device = device
        self.max_models = max_models
        self.access_order = []
    
    def get(self, model_path):
        key = str(model_path)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Load new model with EAGER attention (required for output_attentions)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager"  # Required to get attention weights
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_models:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            torch.cuda.empty_cache()
        
        self.cache[key] = (model, tokenizer)
        self.access_order.append(key)
        
        return model, tokenizer
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        torch.cuda.empty_cache()


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    print("\n" + "="*70)
    print("🔧 FIXED ATTENTION EXTRACTION (v2)")
    print("="*70)
    print(f"   Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = check_gpu()
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check for trained models
    if not DATA_DIR.exists():
        print(f"\n❌ Model directory not found: {DATA_DIR}")
        return
    
    # Load predictions to know which model predicted each case
    preds_path = DATA_DIR / "all_predictions.csv"
    if not preds_path.exists():
        print(f"\n❌ Predictions file not found: {preds_path}")
        return
    
    all_preds = pd.read_csv(preds_path)
    print(f"\n📂 Loaded {len(all_preds)} predictions from all_predictions.csv")
    print(f"   Columns: {list(all_preds.columns)}")
    
    # Load original data
    print(f"\n📂 Loading original data...")
    df = pd.read_pickle(ORIGINAL_DATA)
    outcomes = ["summary judgment granted", "summary judgment refused"]
    df = df[df["outcome"].isin(outcomes)].copy()
    df["label"] = df["outcome"].map({"summary judgment granted": 0, "summary judgment refused": 1})
    df['case_id'] = range(len(df))
    print(f"   Loaded {len(df)} cases")
    
    # Load consistency data
    consistency = pd.read_csv(DATA_DIR / "case_consistency.csv")
    df = df.merge(consistency[['case_id', 'times_correct', 'stratum']], on='case_id', how='left')
    
    # Build model path lookup
    model_paths = {}
    for seed_dir in DATA_DIR.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[1])
        for fold_dir in seed_dir.glob("fold_*"):
            fold = int(fold_dir.name.split("_")[1])
            model_path = fold_dir / "best_model"
            if model_path.exists():
                model_paths[(seed, fold)] = model_path
    
    print(f"   Found {len(model_paths)} trained models")
    
    # Verify label mapping and special tokens from first model
    first_path = list(model_paths.values())[0]
    test_model = AutoModelForSequenceClassification.from_pretrained(
        first_path,
        attn_implementation="eager"  # Required for attention extraction
    )
    test_tokenizer = AutoTokenizer.from_pretrained(first_path)
    
    # Verify special tokens
    tokens_ok = verify_special_tokens(test_tokenizer)
    if not tokens_ok:
        print("\n   ❌ Special tokens not found! Cannot continue.")
        return
    
    # Get label indices
    granted_idx, refused_idx = get_label_indices(test_model)
    print(f"\n📋 Label mapping:")
    print(f"   id2label: {test_model.config.id2label}")
    print(f"   granted_idx: {granted_idx}, refused_idx: {refused_idx}")
    
    del test_model, test_tokenizer
    torch.cuda.empty_cache()
    
    # Initialize model cache
    model_cache = ModelCache(device, max_models=5)
    
    # =========================================================================
    # EXTRACT ATTENTION USING CORRECT MODEL PER CASE
    # =========================================================================
    
    print(f"\n🔬 Extracting attention (using correct model per prediction)...")
    
    results = []
    
    # Group predictions by case for efficiency
    case_predictions = all_preds.groupby('case_id')
    
    for case_id, case_preds in tqdm(case_predictions, total=len(case_predictions), desc="Extracting"):
        case_row = df[df['case_id'] == case_id].iloc[0]
        text = format_text(case_row)
        
        # For each prediction of this case (one per seed)
        for _, pred_row in case_preds.iterrows():
            seed = pred_row['seed']
            fold = pred_row['fold']
            
            model_key = (seed, fold)
            if model_key not in model_paths:
                continue
            
            # Get the correct model
            model, tokenizer = model_cache.get(model_paths[model_key])
            
            # Extract attention
            attn_info = extract_attention(model, tokenizer, text, device)
            metrics = compute_section_metrics(
                attn_info['tokens'], 
                attn_info['cls_attention_mean'],
                attn_info['cls_attention_max']
            )
            
            results.append({
                'case_id': case_id,
                'seed': seed,
                'fold': fold,
                'label': case_row['label'],
                'times_correct': case_row['times_correct'],
                'stratum': case_row['stratum'],
                'correct': pred_row.get('correct', None),
                # RAW DENSITY (mean-based) - THE MEANINGFUL METRIC
                'raw_density_FACTS': metrics['raw_density_mean']['FACTS'],
                'raw_density_APPLICANT': metrics['raw_density_mean']['APPLICANT'],
                'raw_density_DEFENCE': metrics['raw_density_mean']['DEFENCE'],
                # RAW DENSITY (max-based) - may reveal selective heads
                'raw_density_max_FACTS': metrics['raw_density_max']['FACTS'],
                'raw_density_max_APPLICANT': metrics['raw_density_max']['APPLICANT'],
                'raw_density_max_DEFENCE': metrics['raw_density_max']['DEFENCE'],
                # Proportion (artifact)
                'proportion_FACTS': metrics['proportion']['FACTS'],
                'proportion_APPLICANT': metrics['proportion']['APPLICANT'],
                'proportion_DEFENCE': metrics['proportion']['DEFENCE'],
                # Token counts
                'tokens_FACTS': metrics['token_counts']['FACTS'],
                'tokens_APPLICANT': metrics['token_counts']['APPLICANT'],
                'tokens_DEFENCE': metrics['token_counts']['DEFENCE'],
            })
    
    model_cache.clear()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "attention_per_prediction.csv", index=False)
    print(f"   ✅ Saved: {OUTPUT_DIR}/attention_per_prediction.csv")
    
    # Aggregate per case (mean across seeds)
    case_summary = results_df.groupby('case_id').agg({
        'label': 'first',
        'times_correct': 'first',
        'stratum': 'first',
        'raw_density_FACTS': 'mean',
        'raw_density_APPLICANT': 'mean',
        'raw_density_DEFENCE': 'mean',
        'raw_density_max_FACTS': 'mean',
        'raw_density_max_APPLICANT': 'mean',
        'raw_density_max_DEFENCE': 'mean',
        'proportion_FACTS': 'mean',
        'proportion_APPLICANT': 'mean',
        'proportion_DEFENCE': 'mean',
        'tokens_FACTS': 'mean',
        'tokens_APPLICANT': 'mean',
        'tokens_DEFENCE': 'mean',
    }).reset_index()
    
    case_summary.to_csv(OUTPUT_DIR / "attention_per_case.csv", index=False)
    print(f"   ✅ Saved: {OUTPUT_DIR}/attention_per_case.csv")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("📊 ANALYSIS")
    print(f"{'='*70}")
    
    # Verify artifact: proportion ≈ token count ratio
    print(f"\n   TOKEN COUNT RATIOS (explains proportion artifact):")
    total_facts = case_summary['tokens_FACTS'].sum()
    total_app = case_summary['tokens_APPLICANT'].sum()
    total_def = case_summary['tokens_DEFENCE'].sum()
    total_all = total_facts + total_app + total_def
    
    print(f"   FACTS:     {total_facts/total_all:.1%} of tokens")
    print(f"   APPLICANT: {total_app/total_all:.1%} of tokens")
    print(f"   DEFENCE:   {total_def/total_all:.1%} of tokens")
    
    # Proportion (artifact)
    print(f"\n   PROPORTION (artifact - should match token ratios above):")
    print(f"   {'Correct':<10} {'N':<6} {'FACTS':<10} {'APPLICANT':<12} {'DEFENCE':<10}")
    print(f"   {'-'*48}")
    
    for tc in sorted(case_summary['times_correct'].dropna().unique()):
        subset = case_summary[case_summary['times_correct'] == tc]
        print(f"   {int(tc)}/5       {len(subset):<6} "
              f"{subset['proportion_FACTS'].mean():.1%}      "
              f"{subset['proportion_APPLICANT'].mean():.1%}        "
              f"{subset['proportion_DEFENCE'].mean():.1%}")
    
    # Raw density (meaningful - NOT normalized)
    print(f"\n   RAW DENSITY (mean attention per token - NOT normalized):")
    print(f"   {'Correct':<10} {'N':<6} {'FACTS':<12} {'APPLICANT':<14} {'DEFENCE':<12}")
    print(f"   {'-'*54}")
    
    for tc in sorted(case_summary['times_correct'].dropna().unique()):
        subset = case_summary[case_summary['times_correct'] == tc]
        print(f"   {int(tc)}/5       {len(subset):<6} "
              f"{subset['raw_density_FACTS'].mean():.4f}       "
              f"{subset['raw_density_APPLICANT'].mean():.4f}         "
              f"{subset['raw_density_DEFENCE'].mean():.4f}")
    
    # Max-based density
    print(f"\n   RAW DENSITY (max across heads - reveals selective attention):")
    print(f"   {'Correct':<10} {'N':<6} {'FACTS':<12} {'APPLICANT':<14} {'DEFENCE':<12}")
    print(f"   {'-'*54}")
    
    for tc in sorted(case_summary['times_correct'].dropna().unique()):
        subset = case_summary[case_summary['times_correct'] == tc]
        print(f"   {int(tc)}/5       {len(subset):<6} "
              f"{subset['raw_density_max_FACTS'].mean():.4f}       "
              f"{subset['raw_density_max_APPLICANT'].mean():.4f}         "
              f"{subset['raw_density_max_DEFENCE'].mean():.4f}")
    
    # =========================================================================
    # ABLATION STUDY (using correct models per prediction)
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"🔬 ABLATION STUDY (masking with {MASK_LENGTH} [MASK] tokens)")
    print(f"{'='*70}")
    
    # Sample cases stratified by consistency
    sample_ids = []
    for tc in [0, 1, 2, 3, 4, 5]:
        tc_cases = case_summary[case_summary['times_correct'] == tc]['case_id'].tolist()
        sample_ids.extend(tc_cases[:min(40, len(tc_cases))])  # Up to 40 per group
    
    print(f"\n   Running ablation on {len(sample_ids)} sampled cases...")
    print(f"   Using CORRECT model per prediction (aligned with CV results)")
    
    # Reinitialize model cache for ablation
    ablation_cache = ModelCache(device, max_models=5)
    
    ablation_results = []
    
    # Get predictions for sampled cases
    sample_preds = all_preds[all_preds['case_id'].isin(sample_ids)]
    
    for case_id in tqdm(sample_ids, desc="Ablation"):
        case_row = df[df['case_id'] == case_id].iloc[0]
        text = format_text(case_row)
        
        # Get all predictions for this case
        case_preds = sample_preds[sample_preds['case_id'] == case_id]
        
        # Run ablation using each model that predicted this case
        case_deltas = {'facts': [], 'applicant': [], 'defence': []}
        case_flips = {'facts': [], 'applicant': [], 'defence': []}
        
        for _, pred_row in case_preds.iterrows():
            seed = pred_row['seed']
            fold = pred_row['fold']
            
            model_key = (seed, fold)
            if model_key not in model_paths:
                continue
            
            model, tokenizer = ablation_cache.get(model_paths[model_key])
            g_idx, r_idx = get_label_indices(model)
            
            # Baseline
            base = get_prediction_probs(model, tokenizer, text, device, g_idx, r_idx)
            
            # Mask each section
            for section in ['FACTS', 'APPLICANT', 'DEFENCE']:
                masked_text = mask_section(text, section, n_masks=MASK_LENGTH)
                masked = get_prediction_probs(model, tokenizer, masked_text, device, g_idx, r_idx)
                
                case_deltas[section.lower()].append(masked['prob_refused'] - base['prob_refused'])
                case_flips[section.lower()].append(int(masked['pred'] != base['pred']))
        
        # Average across all models that predicted this case
        ablation_results.append({
            'case_id': case_id,
            'label': case_row['label'],
            'times_correct': case_row['times_correct'],
            'n_models': len(case_preds),
            'facts_delta': np.mean(case_deltas['facts']) if case_deltas['facts'] else 0,
            'applicant_delta': np.mean(case_deltas['applicant']) if case_deltas['applicant'] else 0,
            'defence_delta': np.mean(case_deltas['defence']) if case_deltas['defence'] else 0,
            'facts_flipped': np.mean(case_flips['facts']) if case_flips['facts'] else 0,
            'applicant_flipped': np.mean(case_flips['applicant']) if case_flips['applicant'] else 0,
            'defence_flipped': np.mean(case_flips['defence']) if case_flips['defence'] else 0,
        })
    
    ablation_cache.clear()
    
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"   ✅ Saved: {OUTPUT_DIR}/ablation_results.csv")
    
    # Analyze ablation
    print(f"\n   ABLATION IMPACT (Δ P(refused) when section masked):")
    print(f"   {'Correct':<10} {'N':<6} {'FACTS Δ':<12} {'APPLICANT Δ':<14} {'DEFENCE Δ':<12}")
    print(f"   {'-'*54}")
    
    for tc in sorted(ablation_df['times_correct'].dropna().unique()):
        subset = ablation_df[ablation_df['times_correct'] == tc]
        print(f"   {int(tc)}/5       {len(subset):<6} "
              f"{subset['facts_delta'].mean():+.4f}       "
              f"{subset['applicant_delta'].mean():+.4f}         "
              f"{subset['defence_delta'].mean():+.4f}")
    
    print(f"\n   FLIP RATE (% predictions that change when section masked):")
    print(f"   {'Correct':<10} {'N':<6} {'FACTS':<10} {'APPLICANT':<12} {'DEFENCE':<10}")
    print(f"   {'-'*48}")
    
    for tc in sorted(ablation_df['times_correct'].dropna().unique()):
        subset = ablation_df[ablation_df['times_correct'] == tc]
        print(f"   {int(tc)}/5       {len(subset):<6} "
              f"{subset['facts_flipped'].mean():.1%}      "
              f"{subset['applicant_flipped'].mean():.1%}        "
              f"{subset['defence_flipped'].mean():.1%}")
    
    # Compare always-wrong vs always-right
    print(f"\n{'='*70}")
    print("📊 ALWAYS WRONG (0/5) vs ALWAYS RIGHT (5/5)")
    print(f"{'='*70}")
    
    aw = case_summary[case_summary['times_correct'] == 0]
    ar = case_summary[case_summary['times_correct'] == 5]
    
    print(f"\n   RAW DENSITY (mean attention per token):")
    print(f"   {'Group':<15} {'N':<6} {'FACTS':<12} {'APPLICANT':<14} {'DEFENCE':<12}")
    print(f"   {'-'*59}")
    
    if len(aw) > 0:
        print(f"   {'Always Wrong':<15} {len(aw):<6} "
              f"{aw['raw_density_FACTS'].mean():.4f}       "
              f"{aw['raw_density_APPLICANT'].mean():.4f}         "
              f"{aw['raw_density_DEFENCE'].mean():.4f}")
    
    if len(ar) > 0:
        print(f"   {'Always Right':<15} {len(ar):<6} "
              f"{ar['raw_density_FACTS'].mean():.4f}       "
              f"{ar['raw_density_APPLICANT'].mean():.4f}         "
              f"{ar['raw_density_DEFENCE'].mean():.4f}")
    
    # Ablation comparison
    aw_abl = ablation_df[ablation_df['times_correct'] == 0]
    ar_abl = ablation_df[ablation_df['times_correct'] == 5]
    
    print(f"\n   ABLATION IMPACT:")
    print(f"   {'Group':<15} {'N':<6} {'FACTS Δ':<12} {'APPLICANT Δ':<14} {'DEFENCE Δ':<12}")
    print(f"   {'-'*59}")
    
    if len(aw_abl) > 0:
        print(f"   {'Always Wrong':<15} {len(aw_abl):<6} "
              f"{aw_abl['facts_delta'].mean():+.4f}       "
              f"{aw_abl['applicant_delta'].mean():+.4f}         "
              f"{aw_abl['defence_delta'].mean():+.4f}")
    
    if len(ar_abl) > 0:
        print(f"   {'Always Right':<15} {len(ar_abl):<6} "
              f"{ar_abl['facts_delta'].mean():+.4f}       "
              f"{ar_abl['applicant_delta'].mean():+.4f}         "
              f"{ar_abl['defence_delta'].mean():+.4f}")
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\n   Output files in: {OUTPUT_DIR}/")
    print(f"   - attention_per_prediction.csv (raw density per seed/fold)")
    print(f"   - attention_per_case.csv (aggregated per case)")
    print(f"   - ablation_results.csv (masking impact)")
    print(f"\n   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
