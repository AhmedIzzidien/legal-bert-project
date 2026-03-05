#!/usr/bin/env python3
"""
===============================================================================
LEGAL-BERT MULTI-SEED 5-FOLD CV WITH ATTENTION ANALYSIS
===============================================================================

VERSION 4 - 5 SEEDS, LABEL BREAKDOWN, ATTENTION SUMMARIES

Features:
    - 5 seeds × 5 folds = 25 training runs
    - Each case predicted 5 times (once per seed)
    - BREAKDOWN BY LABEL (granted/refused) - class imbalance aware
    - NO HARDCODED THRESHOLDS - explore first
    - ATTENTION: per-row + per-case summary (mean across models)
    - GPU required, checkpointing, resume support

Usage:
    python legalbert_multiseed_cv_attention.py
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ===============================================================================
# GPU CHECK
# ===============================================================================

def check_gpu_or_die():
    """Check GPU is available. Exit if not."""
    
    print("\n" + "="*70)
    print("🔍 GPU CHECK")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n   ❌ CUDA NOT AVAILABLE")
        print("   This script requires a GPU.")
        print("\n   Fix:")
        print("   pip uninstall torch")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("\n   Exiting...")
        exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"\n   ✅ CUDA AVAILABLE")
    print(f"   GPU: {gpu_name}")
    print(f"   Memory: {gpu_memory:.1f} GB")
    
    try:
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        print(f"   Status: Working")
    except Exception as e:
        print(f"\n   ❌ GPU test failed: {e}")
        exit(1)
    
    return True


# ===============================================================================
# CONFIGURATION
# ===============================================================================

@dataclass
class Config:
    """Configuration."""
    
    # Data
    input_dataframe: str = "sj_231025.pkl"
    n_folds: int = 5
    seeds: List[int] = None  # 5 seeds for 5 predictions per case
    
    # Model
    model_name: str = "nlpaueb/legal-bert-base-uncased"
    max_length: int = 512
    token_style: str = "special"
    
    # Training
    use_class_weights: bool = True
    use_upsampling: bool = True
    label_smoothing: float = 0.1
    
    # Hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 15
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 3
    
    # Attention
    extract_attention: bool = True
    attention_layer: int = -1
    
    # Output
    output_dir: str = "legalbert_multiseed_attention"
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1011]  # 5 seeds


# ===============================================================================
# CHECKPOINT MANAGEMENT
# ===============================================================================

def get_checkpoint_path(config: Config) -> Path:
    return Path(config.output_dir) / "checkpoint_progress.json"


def save_checkpoint(config: Config, completed_folds: List[Tuple[int, int]], 
                    all_predictions: List[pd.DataFrame], all_fold_metrics: List[Dict]):
    checkpoint_path = get_checkpoint_path(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    if all_predictions:
        preds_df = pd.concat(all_predictions, ignore_index=True)
        preds_df.to_csv(Path(config.output_dir) / "checkpoint_predictions.csv", index=False)
    
    checkpoint = {
        'completed_folds': completed_folds,
        'all_fold_metrics': all_fold_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"   💾 Checkpoint: {len(completed_folds)} folds done")


def load_checkpoint(config: Config) -> Tuple[List[Tuple[int, int]], List[pd.DataFrame], List[Dict]]:
    checkpoint_path = get_checkpoint_path(config)
    
    if not checkpoint_path.exists():
        return [], [], []
    
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    completed_folds = [tuple(x) for x in checkpoint.get('completed_folds', [])]
    all_fold_metrics = checkpoint.get('all_fold_metrics', [])
    
    preds_path = Path(config.output_dir) / "checkpoint_predictions.csv"
    if preds_path.exists():
        preds_df = pd.read_csv(preds_path)
        all_predictions = [
            preds_df[(preds_df['seed'] == seed) & (preds_df['fold'] == fold)].copy()
            for seed, fold in completed_folds
        ]
    else:
        all_predictions = []
    
    print(f"\n   📂 Resuming: {len(completed_folds)} folds already done")
    return completed_folds, all_predictions, all_fold_metrics


def is_fold_completed(seed: int, fold: int, completed_folds: List[Tuple[int, int]]) -> bool:
    return (seed, fold) in completed_folds


# ===============================================================================
# TEXT & DATA
# ===============================================================================

TOKEN_MARKERS = {
    "special": {"facts": "[FACTS]", "applicant": "[APPLICANT]", "defence": "[DEFENCE]"},
    "natural": {"facts": "Facts of the case:", "applicant": "Applicant submits:", "defence": "Defence submits:"},
}


def format_text(row: pd.Series, token_style: str = "special") -> str:
    markers = TOKEN_MARKERS[token_style]
    parts = []
    
    for field, marker in [('facts', 'facts'), ('applicant_reason', 'applicant'), ('defence_reason', 'defence')]:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            parts.append(f"{markers[marker]} {str(row[field]).strip()}")
    
    return " ".join(parts)


def parse_let_categories(df: pd.DataFrame) -> pd.DataFrame:
    cat_col = None
    for col in ['decision_reason_categories_clean', 'decision_reason_categories']:
        if col in df.columns:
            cat_col = col
            break
    
    if cat_col is None:
        df['LAW'] = df['EVIDENCE'] = df['TRIAL'] = None
        return df
    
    def parse_cats(cat_str):
        if pd.isna(cat_str):
            return {'LAW': None, 'EVIDENCE': None, 'TRIAL': None}
        cats = {}
        for item in str(cat_str).split():
            if '=' in item:
                key, val = item.split('=')
                cats[key] = int(val) if val.isdigit() else None
        return cats
    
    df['_cats'] = df[cat_col].apply(parse_cats)
    df['LAW'] = df['_cats'].apply(lambda x: x.get('LAW'))
    df['EVIDENCE'] = df['_cats'].apply(lambda x: x.get('EVIDENCE'))
    df['TRIAL'] = df['_cats'].apply(lambda x: x.get('TRIAL'))
    return df


def get_stratum(row) -> str:
    l, e, t = row.get('LAW'), row.get('EVIDENCE'), row.get('TRIAL')
    
    if l == 1 and e == 1 and t == 1: return 'ALL THREE'
    elif l == 1 and e == 0 and t == 1: return 'LAW + TRIAL'
    elif l == 1 and e == 1 and t == 0: return 'LAW + EVIDENCE'
    elif l == 0 and e == 1 and t == 1: return 'EVIDENCE + TRIAL'
    elif l == 1 and e == 0 and t == 0: return 'LAW only'
    elif l == 0 and e == 1 and t == 0: return 'EVIDENCE only'
    elif l == 0 and e == 0 and t == 1: return 'TRIAL only'
    return 'UNKNOWN'


# ===============================================================================
# DATASET & TRAINER
# ===============================================================================

class LegalDataset(Dataset):
    def __init__(self, texts, labels, case_ids, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.case_ids = case_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'case_id': self.case_ids[idx],
        }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("case_id", None)
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_granted': f1_score(labels, predictions, pos_label=0),
        'f1_refused': f1_score(labels, predictions, pos_label=1),
    }


def upsample_minority(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    majority = df[df['label'] == 0]
    minority = df[df['label'] == 1]
    minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=random_seed)
    upsampled = pd.concat([majority, minority_upsampled])
    return upsampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)


# ===============================================================================
# TRAINING
# ===============================================================================

def train_fold(fold, seed, train_df, val_df, config, tokenizer) -> pd.DataFrame:
    print(f"      Fold {fold + 1}: Train={len(train_df)}, Val={len(val_df)}")
    
    if config.use_upsampling:
        train_df = upsample_minority(train_df, seed + fold)
    
    n_granted = (train_df['label'] == 0).sum()
    n_refused = (train_df['label'] == 1).sum()
    total = len(train_df)
    
    class_weights = torch.tensor([
        total / (2 * n_granted), total / (2 * n_refused)
    ], dtype=torch.float32) if config.use_class_weights else None
    
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    if config.token_style == "special":
        model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device('cuda')
    model = model.to(device)
    
    train_dataset = LegalDataset(
        train_df['text'].tolist(), train_df['label'].tolist(),
        train_df['case_id'].tolist(), tokenizer, config.max_length
    )
    val_dataset = LegalDataset(
        val_df['text'].tolist(), val_df['label'].tolist(),
        val_df['case_id'].tolist(), tokenizer, config.max_length
    )
    
    fold_dir = Path(config.output_dir) / f"seed_{seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(fold_dir / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="no",
        save_total_limit=1,
        fp16=True,
        report_to="none",
    )
    
    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights, label_smoothing=config.label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    
    trainer.train()
    
    model_save_path = fold_dir / "best_model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    print(f"      💾 Model: {model_save_path}")
    
    model = trainer.model.to(device)
    model.eval()
    
    predictions = []
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            case_ids = batch['case_id']
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            for i in range(len(case_ids)):
                predictions.append({
                    'case_id': case_ids[i].item() if torch.is_tensor(case_ids[i]) else case_ids[i],
                    'label': labels[i].item(),
                    'predicted': preds[i].item(),
                    'prob_granted': probs[i, 0].item(),
                    'prob_refused': probs[i, 1].item(),
                    'correct': labels[i].item() == preds[i].item(),
                    'seed': seed,
                    'fold': fold,
                })
    
    fold_preds_df = pd.DataFrame(predictions)
    fold_preds_df.to_csv(fold_dir / "predictions.csv", index=False)
    
    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()
    
    return fold_preds_df


# ===============================================================================
# ATTENTION EXTRACTION
# ===============================================================================

def extract_attention_for_case(model, tokenizer, text, config):
    model.eval()
    device = next(model.parameters()).device
    
    encoding = tokenizer(text, truncation=True, max_length=config.max_length,
                         padding='max_length', return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    
    attention = outputs.attentions[config.attention_layer]
    attention_avg = attention.mean(dim=1).squeeze(0)
    cls_attention = attention_avg[0, :].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    seq_len = attention_mask.sum().item()
    
    return {'tokens': tokens[:seq_len], 'cls_attention': cls_attention[:seq_len]}


def find_section_attention(tokens, attention):
    section_attention = {'FACTS': 0.0, 'APPLICANT': 0.0, 'DEFENCE': 0.0, 'OTHER': 0.0}
    current_section = 'OTHER'
    
    for token, attn in zip(tokens, attention):
        if token == '[FACTS]': current_section = 'FACTS'
        elif token == '[APPLICANT]': current_section = 'APPLICANT'
        elif token == '[DEFENCE]': current_section = 'DEFENCE'
        elif token in ['[CLS]', '[SEP]', '[PAD]']: continue
        else: section_attention[current_section] += attn
    
    total = sum(section_attention.values())
    if total > 0:
        section_attention = {k: v / total for k, v in section_attention.items()}
    return section_attention


def get_top_tokens(tokens, attention, top_k=10):
    valid_pairs = [(t, a) for t, a in zip(tokens, attention)
                   if t not in ['[CLS]', '[SEP]', '[PAD]', '[FACTS]', '[APPLICANT]', '[DEFENCE]']]
    return sorted(valid_pairs, key=lambda x: x[1], reverse=True)[:top_k]


def extract_all_attention(all_preds_df, df_full, config, tokenizer):
    """Extract attention from all models that predicted each case."""
    
    print(f"\n{'='*70}")
    print("🔍 EXTRACTING ATTENTION")
    print(f"{'='*70}")
    
    device = torch.device('cuda')
    n_seeds = len(config.seeds)
    
    case_to_models = defaultdict(list)
    for _, row in all_preds_df.iterrows():
        case_to_models[row['case_id']].append((row['seed'], row['fold'], row['correct']))
    
    loaded_models = {}
    attention_rows = []  # Per-row (case × seed)
    
    unique_cases = df_full['case_id'].unique()
    
    for case_id in tqdm(unique_cases, desc="   Extracting"):
        case_data = df_full[df_full['case_id'] == case_id].iloc[0]
        text = case_data['text']
        
        models_for_case = case_to_models.get(case_id, [])
        if not models_for_case:
            continue
        
        for seed, fold, correct in models_for_case:
            model_key = (seed, fold)
            
            if model_key not in loaded_models:
                model_path = Path(config.output_dir) / f"seed_{seed}" / f"fold_{fold}" / "best_model"
                if not model_path.exists():
                    continue
                
                model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
                model = model.to(device)
                model.eval()
                loaded_models[model_key] = model
                
                if len(loaded_models) > 5:
                    old_key = list(loaded_models.keys())[0]
                    del loaded_models[old_key]
                    torch.cuda.empty_cache()
            
            model = loaded_models[model_key]
            attn_info = extract_attention_for_case(model, tokenizer, text, config)
            section_attn = find_section_attention(attn_info['tokens'], attn_info['cls_attention'])
            top_tokens = get_top_tokens(attn_info['tokens'], attn_info['cls_attention'], top_k=5)
            
            attention_rows.append({
                'case_id': case_id,
                'seed': seed,
                'fold': fold,
                'label': case_data['label'],
                'correct': correct,
                'stratum': case_data.get('stratum', 'UNKNOWN'),
                'attn_FACTS': section_attn['FACTS'],
                'attn_APPLICANT': section_attn['APPLICANT'],
                'attn_DEFENCE': section_attn['DEFENCE'],
                'top_tokens': ', '.join([f"{t}({a:.3f})" for t, a in top_tokens]),
            })
    
    for model in loaded_models.values():
        del model
    torch.cuda.empty_cache()
    
    return pd.DataFrame(attention_rows)


def create_attention_summary(attention_df, config):
    """Create per-case attention summary (mean across seeds)."""
    
    n_seeds = len(config.seeds)
    
    summary = attention_df.groupby('case_id').agg({
        'label': 'first',
        'stratum': 'first',
        'correct': ['sum', 'count'],
        'attn_FACTS': 'mean',
        'attn_APPLICANT': 'mean',
        'attn_DEFENCE': 'mean',
        'top_tokens': list,
    }).reset_index()
    
    summary.columns = [
        'case_id', 'label', 'stratum', 'times_correct', 'times_seen',
        'mean_attn_FACTS', 'mean_attn_APPLICANT', 'mean_attn_DEFENCE', 'top_tokens_list'
    ]
    
    summary['times_wrong'] = summary['times_seen'] - summary['times_correct']
    
    return summary


# ===============================================================================
# CONSISTENCY ANALYSIS
# ===============================================================================

def analyze_consistency(all_preds_df, df_full, config):
    """Analyze consistency with breakdown by label."""
    
    n_seeds = len(config.seeds)
    
    case_stats = all_preds_df.groupby('case_id').agg({
        'correct': ['sum', 'count', list],
        'label': 'first',
        'predicted': list,
        'prob_granted': 'mean',
        'prob_refused': 'mean',
        'seed': list,
    }).reset_index()
    
    case_stats.columns = [
        'case_id', 'times_correct', 'times_seen', 'correct_list',
        'label', 'predicted_list', 'avg_prob_granted', 'avg_prob_refused', 'seed_list'
    ]
    
    stratum_map = df_full.set_index('case_id')['stratum'].to_dict()
    case_stats['stratum'] = case_stats['case_id'].map(stratum_map)
    case_stats['times_wrong'] = case_stats['times_seen'] - case_stats['times_correct']
    case_stats['label_name'] = case_stats['label'].map({0: 'Granted', 1: 'Refused'})
    
    def get_wrong_seeds(row):
        return [s for s, c in zip(row['seed_list'], row['correct_list']) if not c]
    
    case_stats['wrong_seeds'] = case_stats.apply(get_wrong_seeds, axis=1)
    case_stats['wrong_seeds_str'] = case_stats['wrong_seeds'].apply(
        lambda x: ','.join(map(str, x)) if x else ''
    )
    
    return case_stats


def print_distribution(case_stats, config, title="OVERALL"):
    """Print consistency distribution."""
    
    n_seeds = len(config.seeds)
    
    print(f"\n   {title}:")
    print(f"   {'Correct':<12} {'Count':<10} {'Pct':<10}")
    print(f"   {'-'*32}")
    
    dist = case_stats['times_correct'].value_counts().sort_index()
    for tc, count in dist.items():
        pct = count / len(case_stats) * 100
        print(f"   {tc}/{n_seeds}          {count:<10} {pct:.1f}%")
    
    return dist


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    print("\n" + "="*70)
    print("🔬 LEGAL-BERT 5-SEED CV WITH ATTENTION")
    print("="*70)
    print(f"   Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_gpu_or_die()
    
    config = Config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_seeds = len(config.seeds)
    print(f"\n   Seeds: {config.seeds}")
    print(f"   Folds: {config.n_folds}")
    print(f"   Total runs: {n_seeds * config.n_folds}")
    print(f"   Predictions per case: {n_seeds}")
    
    completed_folds, all_predictions, all_fold_metrics = load_checkpoint(config)
    
    if not Path(config.input_dataframe).exists():
        print(f"\n   ❌ Data not found: {config.input_dataframe}")
        return
    
    print(f"\n📂 Loading data...")
    df = pd.read_pickle(config.input_dataframe)
    
    outcomes = ["summary judgment granted", "summary judgment refused"]
    df = df[df["outcome"].isin(outcomes)].copy()
    
    df["label"] = df["outcome"].map({"summary judgment granted": 0, "summary judgment refused": 1})
    df["text"] = df.apply(lambda row: format_text(row, config.token_style), axis=1)
    df = parse_let_categories(df)
    df['stratum'] = df.apply(get_stratum, axis=1)
    df['case_id'] = range(len(df))
    
    n_granted = (df['label'] == 0).sum()
    n_refused = (df['label'] == 1).sum()
    print(f"   Total: {len(df)} (Granted: {n_granted}, Refused: {n_refused})")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.token_style == "special":
        tokenizer.add_special_tokens({'additional_special_tokens': ['[FACTS]', '[APPLICANT]', '[DEFENCE]']})
    
    # Training
    for seed in config.seeds:
        print(f"\n{'='*70}")
        print(f"🎲 SEED {seed}")
        print(f"{'='*70}")
        
        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            if is_fold_completed(seed, fold, completed_folds):
                print(f"      Fold {fold + 1}: ⏭️ Skipping")
                continue
            
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            fold_preds = train_fold(fold, seed, train_df, val_df, config, tokenizer)
            all_predictions.append(fold_preds)
            
            acc = fold_preds['correct'].mean()
            f1 = f1_score(fold_preds['label'], fold_preds['predicted'], average='macro')
            
            all_fold_metrics.append({'seed': seed, 'fold': fold, 'accuracy': acc, 'f1_macro': f1})
            completed_folds.append((seed, fold))
            save_checkpoint(config, completed_folds, all_predictions, all_fold_metrics)
            
            print(f"      Fold {fold + 1}: Acc={acc:.1%}, F1={f1:.3f} ✅")
    
    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    
    # Consistency Analysis
    print(f"\n{'='*70}")
    print("📊 CONSISTENCY ANALYSIS")
    print(f"{'='*70}")
    
    case_stats = analyze_consistency(all_preds_df, df, config)
    
    # Overall distribution
    overall_dist = print_distribution(case_stats, config, "OVERALL DISTRIBUTION")
    
    # By label
    granted_cases = case_stats[case_stats['label'] == 0]
    refused_cases = case_stats[case_stats['label'] == 1]
    
    print(f"\n   BY LABEL:")
    granted_dist = print_distribution(granted_cases, config, f"GRANTED (n={len(granted_cases)})")
    refused_dist = print_distribution(refused_cases, config, f"REFUSED (n={len(refused_cases)})")
    
    # By stratum for 0/N correct
    always_wrong = case_stats[case_stats['times_correct'] == 0]
    if len(always_wrong) > 0:
        print(f"\n   ALWAYS WRONG (0/{n_seeds}) by STRATUM:")
        for stratum, count in always_wrong['stratum'].value_counts().items():
            print(f"      {stratum:<20}: {count}")
    
    # Save case stats
    case_stats.to_csv(output_dir / "case_consistency.csv", index=False)
    print(f"\n   ✅ Saved: case_consistency.csv")
    
    # Distribution tables
    dist_overall = pd.DataFrame({
        'times_correct': overall_dist.index,
        'count': overall_dist.values,
        'pct': overall_dist.values / len(case_stats) * 100
    })
    dist_overall.to_csv(output_dir / "distribution_overall.csv", index=False)
    
    dist_by_label = []
    for label, label_name in [(0, 'Granted'), (1, 'Refused')]:
        subset = case_stats[case_stats['label'] == label]
        dist = subset['times_correct'].value_counts().sort_index()
        for tc, count in dist.items():
            dist_by_label.append({
                'label': label_name,
                'times_correct': tc,
                'count': count,
                'pct': count / len(subset) * 100
            })
    pd.DataFrame(dist_by_label).to_csv(output_dir / "distribution_by_label.csv", index=False)
    
    print(f"   ✅ Saved: distribution_overall.csv")
    print(f"   ✅ Saved: distribution_by_label.csv")
    
    # Attention
    if config.extract_attention:
        attention_df = extract_all_attention(all_preds_df, df, config, tokenizer)
        
        if len(attention_df) > 0:
            # Per-row attention
            attention_df.to_csv(output_dir / "attention_per_row.csv", index=False)
            print(f"   ✅ Saved: attention_per_row.csv")
            
            # Per-case summary
            attention_summary = create_attention_summary(attention_df, config)
            attention_summary.to_csv(output_dir / "attention_per_case.csv", index=False)
            print(f"   ✅ Saved: attention_per_case.csv")
            
            # Attention summary by consistency
            print(f"\n{'='*70}")
            print("🔍 ATTENTION BY CONSISTENCY")
            print(f"{'='*70}")
            
            print(f"\n   {'Correct':<10} {'N':<8} {'FACTS':<10} {'APPLICANT':<12} {'DEFENCE':<10}")
            print(f"   {'-'*50}")
            
            for tc in sorted(attention_summary['times_correct'].unique()):
                subset = attention_summary[attention_summary['times_correct'] == tc]
                print(f"   {tc}/{n_seeds}        {len(subset):<8} "
                      f"{subset['mean_attn_FACTS'].mean():.1%}      "
                      f"{subset['mean_attn_APPLICANT'].mean():.1%}        "
                      f"{subset['mean_attn_DEFENCE'].mean():.1%}")
    
    # Save all predictions
    all_preds_df.to_csv(output_dir / "all_predictions.csv", index=False)
    pd.DataFrame(all_fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    
    # Summary
    metrics_df = pd.DataFrame(all_fold_metrics)
    
    print(f"\n{'='*70}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n   METRICS ({n_seeds} seeds × {config.n_folds} folds):")
    print(f"   Accuracy:  {metrics_df['accuracy'].mean():.1%} ± {metrics_df['accuracy'].std():.1%}")
    print(f"   F1-Macro:  {metrics_df['f1_macro'].mean():.3f} ± {metrics_df['f1_macro'].std():.3f}")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_seeds': n_seeds,
        'n_folds': config.n_folds,
        'total_cases': len(case_stats),
        'n_granted': int(n_granted),
        'n_refused': int(n_refused),
        'metrics': {
            'accuracy_mean': float(metrics_df['accuracy'].mean()),
            'accuracy_std': float(metrics_df['accuracy'].std()),
            'f1_macro_mean': float(metrics_df['f1_macro'].mean()),
            'f1_macro_std': float(metrics_df['f1_macro'].std()),
        },
        'distribution_overall': {f"{k}/{n_seeds}": int(v) for k, v in overall_dist.items()},
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Cleanup
    for f in [get_checkpoint_path(config), output_dir / "checkpoint_predictions.csv"]:
        if f.exists():
            f.unlink()
    
    print(f"\n   ✅ All saved to: {output_dir}/")
    print(f"\n   Key files:")
    print(f"      - case_consistency.csv       (per-case breakdown)")
    print(f"      - distribution_overall.csv   (0/5, 1/5, ... 5/5)")
    print(f"      - distribution_by_label.csv  (granted vs refused)")
    print(f"      - attention_per_row.csv      (each case × each seed)")
    print(f"      - attention_per_case.csv     (mean across seeds)")
    
    print(f"\n   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
