#!/usr/bin/env python3
"""
===============================================================================
LEGAL-BERT 5-FOLD CROSS-VALIDATION
===============================================================================

Runs 5-fold stratified CV to get stable performance estimates.
Uses SPECIAL tokens format: [FACTS], [APPLICANT], [DEFENCE]

Reports:
    - Overall metrics (mean ± std across folds)
    - L/E/T strata performance (mean ± std across folds)
    - Per-fold breakdown

Usage:
    python legalbert_5fold_cv.py
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
)

import gc

warnings.filterwarnings('ignore')


# ===============================================================================
# CONFIGURATION
# ===============================================================================

@dataclass
class Config:
    """Configuration for 5-fold CV."""
    
    # Data
    input_dataframe: str = "sj_231025.pkl"
    n_folds: int = 5
    random_seed: int = 42
    
    # Model
    model_name: str = "nlpaueb/legal-bert-base-uncased"
    max_length: int = 512
    
    # Token style
    token_style: str = "special"  # "special" or "natural"
    
    # Training
    use_class_weights: bool = True
    use_upsampling: bool = True
    label_smoothing: float = 0.1
    
    # Hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 15  # Fewer epochs per fold for speed
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Early stopping
    early_stopping_patience: int = 3  # Faster for CV
    
    # Output
    output_dir: str = "legalbert_5fold_cv"


# ===============================================================================
# TEXT FORMATTING
# ===============================================================================

TOKEN_MARKERS = {
    "special": {
        "facts": "[FACTS]",
        "applicant": "[APPLICANT]",
        "defence": "[DEFENCE]",
    },
    "natural": {
        "facts": "Facts of the case:",
        "applicant": "Applicant submits:",
        "defence": "Defence submits:",
    },
}


def format_text(row: pd.Series, token_style: str = "special") -> str:
    """Format text with structural tokens."""
    markers = TOKEN_MARKERS[token_style]
    parts = []
    
    if 'facts' in row and pd.notna(row['facts']) and str(row['facts']).strip():
        parts.append(f"{markers['facts']} {str(row['facts']).strip()}")
    
    if 'applicant_reason' in row and pd.notna(row['applicant_reason']) and str(row['applicant_reason']).strip():
        parts.append(f"{markers['applicant']} {str(row['applicant_reason']).strip()}")
    
    if 'defence_reason' in row and pd.notna(row['defence_reason']) and str(row['defence_reason']).strip():
        parts.append(f"{markers['defence']} {str(row['defence_reason']).strip()}")
    
    return " ".join(parts)


# ===============================================================================
# DATASET
# ===============================================================================

class LegalDataset(Dataset):
    """Dataset for Legal-BERT."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ===============================================================================
# CUSTOM TRAINER
# ===============================================================================

class WeightedTrainer(Trainer):
    """Trainer with class weights and label smoothing."""
    
    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ===============================================================================
# METRICS
# ===============================================================================

def compute_metrics(eval_pred):
    """Compute all metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_granted': f1_score(labels, predictions, pos_label=0),
        'f1_refused': f1_score(labels, predictions, pos_label=1),
    }


def upsample_minority(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """Upsample minority class."""
    majority = df[df['label'] == 0]
    minority = df[df['label'] == 1]
    
    minority_upsampled = minority.sample(
        n=len(majority),
        replace=True,
        random_state=random_seed
    )
    
    upsampled = pd.concat([majority, minority_upsampled])
    return upsampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)


# ===============================================================================
# L/E/T STRATA ANALYSIS
# ===============================================================================

def parse_let_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Parse L/E/T categories from decision_reason_categories_clean."""
    
    cat_col = None
    for col in ['decision_reason_categories_clean', 'decision_reason_categories']:
        if col in df.columns:
            cat_col = col
            break
    
    if cat_col is None:
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


def compute_strata_metrics(df: pd.DataFrame) -> Dict:
    """Compute metrics for each L/E/T stratum."""
    
    strata = [
        ('ALL THREE', (df['LAW'] == 1) & (df['EVIDENCE'] == 1) & (df['TRIAL'] == 1)),
        ('LAW + TRIAL', (df['LAW'] == 1) & (df['EVIDENCE'] == 0) & (df['TRIAL'] == 1)),
        ('LAW + EVIDENCE', (df['LAW'] == 1) & (df['EVIDENCE'] == 1) & (df['TRIAL'] == 0)),
        ('EVIDENCE + TRIAL', (df['LAW'] == 0) & (df['EVIDENCE'] == 1) & (df['TRIAL'] == 1)),
        ('LAW only', (df['LAW'] == 1) & (df['EVIDENCE'] == 0) & (df['TRIAL'] == 0)),
        ('EVIDENCE only', (df['LAW'] == 0) & (df['EVIDENCE'] == 1) & (df['TRIAL'] == 0)),
        ('TRIAL only', (df['LAW'] == 0) & (df['EVIDENCE'] == 0) & (df['TRIAL'] == 1)),
    ]
    
    results = {}
    
    for name, mask in strata:
        subset = df[mask]
        if len(subset) < 2:
            results[name] = {'n': len(subset), 'accuracy': None, 'f1_macro': None}
            continue
        
        y_true = subset['label'].values
        y_pred = subset['predicted'].values
        
        results[name] = {
            'n': len(subset),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }
    
    return results


# ===============================================================================
# SINGLE FOLD TRAINING
# ===============================================================================

def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config,
    tokenizer,
) -> Tuple[Dict, pd.DataFrame]:
    """Train a single fold."""
    
    print(f"\n   {'='*60}")
    print(f"   FOLD {fold + 1}/{config.n_folds}")
    print(f"   {'='*60}")
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Upsample training data
    if config.use_upsampling:
        train_df = upsample_minority(train_df, config.random_seed + fold)
    
    # Compute class weights from original distribution
    n_granted = (train_df['label'] == 0).sum()
    n_refused = (train_df['label'] == 1).sum()
    total = len(train_df)
    
    if config.use_class_weights:
        class_weights = torch.tensor([
            total / (2 * n_granted),
            total / (2 * n_refused)
        ], dtype=torch.float32)
    else:
        class_weights = None
    
    # Load fresh model for each fold
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2
    )
    
    # Add special tokens if needed
    if config.token_style == "special":
        model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = LegalDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        config.max_length
    )
    
    val_dataset = LegalDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        config.max_length
    )
    
    # Training arguments
    fold_output_dir = Path(config.output_dir) / f"fold_{fold}"
    
    training_args = TrainingArguments(
        output_dir=str(fold_output_dir / "checkpoints"),
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
        logging_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=config.label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Generate predictions
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    
    val_df = val_df.copy().reset_index(drop=True)
    val_df['predicted'] = preds
    val_df['prob_granted'] = probs[:, 0]
    val_df['prob_refused'] = probs[:, 1]
    val_df['correct'] = val_df['label'] == val_df['predicted']
    
    # Clean up
    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()
    
    return eval_results, val_df


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    """Run 5-fold cross-validation."""
    
    print("\n" + "="*70)
    print("🔬 LEGAL-BERT 5-FOLD CROSS-VALIDATION")
    print("="*70)
    print(f"   Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    config = Config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data file
    if not Path(config.input_dataframe).exists():
        print(f"\n   ❌ ERROR: Data file not found: {config.input_dataframe}")
        return
    
    # Load data
    print(f"\n📂 Loading data from: {config.input_dataframe}")
    df = pd.read_pickle(config.input_dataframe)
    
    # Filter to binary outcomes
    outcomes = ["summary judgment granted", "summary judgment refused"]
    df = df[df["outcome"].isin(outcomes)].copy()
    print(f"   Total cases: {len(df)}")
    
    # Create label
    df["label"] = df["outcome"].map({
        "summary judgment granted": 0,
        "summary judgment refused": 1,
    })
    
    # Format text
    print(f"   Token style: {config.token_style}")
    df["text"] = df.apply(lambda row: format_text(row, config.token_style), axis=1)
    
    # Parse L/E/T categories
    df = parse_let_categories(df)
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.token_style == "special":
        special_tokens = {'additional_special_tokens': ['[FACTS]', '[APPLICANT]', '[DEFENCE]']}
        tokenizer.add_special_tokens(special_tokens)
        print(f"   Added special tokens: [FACTS], [APPLICANT], [DEFENCE]")
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    
    # Storage for results
    all_fold_results = []
    all_fold_strata = []
    all_predictions = []
    
    # Run each fold
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING {config.n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        eval_results, val_df_with_preds = train_fold(fold, train_df, val_df, config, tokenizer)
        
        # Store overall results
        fold_metrics = {
            'fold': fold + 1,
            'accuracy': eval_results['eval_accuracy'],
            'f1_macro': eval_results['eval_f1_macro'],
            'f1_granted': eval_results['eval_f1_granted'],
            'f1_refused': eval_results['eval_f1_refused'],
        }
        all_fold_results.append(fold_metrics)
        
        # Compute strata metrics
        strata_metrics = compute_strata_metrics(val_df_with_preds)
        strata_metrics['fold'] = fold + 1
        all_fold_strata.append(strata_metrics)
        
        # Store predictions
        val_df_with_preds['fold'] = fold + 1
        all_predictions.append(val_df_with_preds)
        
        # Print fold results
        print(f"\n   Fold {fold + 1} Results:")
        print(f"      Accuracy: {eval_results['eval_accuracy']:.1%}")
        print(f"      F1-Macro: {eval_results['eval_f1_macro']:.3f}")
        
        if 'ALL THREE' in strata_metrics and strata_metrics['ALL THREE']['accuracy'] is not None:
            print(f"      ALL THREE: {strata_metrics['ALL THREE']['accuracy']:.1%} (n={strata_metrics['ALL THREE']['n']})")
    
    # Aggregate results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS (Mean ± Std across folds)")
    print("="*70)
    
    results_df = pd.DataFrame(all_fold_results)
    
    print(f"\n   OVERALL METRICS:")
    print(f"   {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print(f"   {'-'*42}")
    for metric in ['accuracy', 'f1_macro', 'f1_granted', 'f1_refused']:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        print(f"   {metric:<20} {mean:>9.3f} {std:>9.3f}")
    
    # Aggregate strata results
    print(f"\n   L/E/T STRATA METRICS:")
    print(f"   {'Category':<20} {'N (avg)':>8} {'Acc Mean':>10} {'Acc Std':>10}")
    print(f"   {'-'*50}")
    
    strata_names = ['ALL THREE', 'LAW + TRIAL', 'LAW + EVIDENCE', 'EVIDENCE + TRIAL', 
                    'LAW only', 'EVIDENCE only', 'TRIAL only']
    
    strata_summary = []
    for stratum in strata_names:
        accs = []
        ns = []
        for fold_strata in all_fold_strata:
            if stratum in fold_strata and fold_strata[stratum]['accuracy'] is not None:
                accs.append(fold_strata[stratum]['accuracy'])
                ns.append(fold_strata[stratum]['n'])
        
        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_n = np.mean(ns)
            print(f"   {stratum:<20} {mean_n:>7.1f} {mean_acc:>9.1%} {std_acc:>9.1%}")
            strata_summary.append({
                'category': stratum,
                'n_mean': mean_n,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
            })
    
    # Save results
    results_df.to_csv(output_dir / "fold_results.csv", index=False)
    print(f"\n   ✅ Saved: fold_results.csv")
    
    pd.DataFrame(strata_summary).to_csv(output_dir / "strata_summary.csv", index=False)
    print(f"   ✅ Saved: strata_summary.csv")
    
    # Save all predictions
    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_df.to_csv(output_dir / "all_predictions.csv", index=False)
    print(f"   ✅ Saved: all_predictions.csv")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'overall_results': {
            'accuracy': {'mean': results_df['accuracy'].mean(), 'std': results_df['accuracy'].std()},
            'f1_macro': {'mean': results_df['f1_macro'].mean(), 'std': results_df['f1_macro'].std()},
            'f1_granted': {'mean': results_df['f1_granted'].mean(), 'std': results_df['f1_granted'].std()},
            'f1_refused': {'mean': results_df['f1_refused'].mean(), 'std': results_df['f1_refused'].std()},
        },
        'strata_summary': strata_summary,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   ✅ Saved: summary.json")
    
    # Plot fold results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 scores by fold
    ax = axes[0]
    x = range(1, config.n_folds + 1)
    ax.plot(x, results_df['f1_macro'], 'b-o', label='F1-Macro')
    ax.plot(x, results_df['f1_granted'], 'g-o', label='F1-Granted')
    ax.plot(x, results_df['f1_refused'], 'r-o', label='F1-Refused')
    ax.axhline(results_df['f1_macro'].mean(), color='b', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores by Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ALL THREE accuracy by fold
    ax = axes[1]
    all_three_accs = [fs['ALL THREE']['accuracy'] for fs in all_fold_strata if fs['ALL THREE']['accuracy'] is not None]
    all_three_ns = [fs['ALL THREE']['n'] for fs in all_fold_strata if fs['ALL THREE']['accuracy'] is not None]
    ax.bar(range(1, len(all_three_accs) + 1), all_three_accs, color='steelblue')
    ax.axhline(np.mean(all_three_accs), color='red', linestyle='--', label=f'Mean: {np.mean(all_three_accs):.1%}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('ALL THREE (L+E+T) Accuracy by Fold')
    ax.legend()
    ax.set_ylim(0, 1)
    
    for i, (acc, n) in enumerate(zip(all_three_accs, all_three_ns)):
        ax.text(i + 1, acc + 0.02, f'n={n}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: cv_results.png")
    
    print(f"\n   📁 All outputs saved to: {output_dir}/")
    print(f"\n   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
