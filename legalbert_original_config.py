#!/usr/bin/env python3
"""
===============================================================================
LEGAL-BERT TRAINING - ORIGINAL CONFIGURATION
===============================================================================

This replicates the EXACT configuration that achieved:
    - 90.7% accuracy on ALL THREE (L+E+T) cases
    - F1-Macro: ~0.657
    - Overall accuracy: 66.8%

Key differences from structural token experiments:
    1. Text format: facts [SEP] applicant_reason [SEP] defence_reason
    2. Class weights calculated from ORIGINAL distribution (before upsampling)
       Granted=0.82, Refused=1.28

Usage:
    python legalbert_original_config.py
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
)

warnings.filterwarnings('ignore')


# ===============================================================================
# CONFIGURATION - ORIGINAL SETTINGS
# ===============================================================================

@dataclass
class Config:
    """Original configuration that achieved 90.7% on ALL THREE."""
    
    # Data
    input_dataframe: str = "sj_231025.pkl"
    val_split: float = 0.25
    random_seed: int = 42
    
    # Model
    model_name: str = "nlpaueb/legal-bert-base-uncased"
    max_length: int = 512
    
    # Training - Original refined approach
    use_class_weights: bool = True
    use_upsampling: bool = True
    label_smoothing: float = 0.1
    
    # CRITICAL: Calculate class weights from ORIGINAL distribution, not after upsampling
    calculate_weights_before_upsampling: bool = True
    
    # Hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Output
    output_dir: str = "legalbert_output_original"


# ===============================================================================
# ORIGINAL TEXT FORMAT
# ===============================================================================

def format_text_original(row: pd.Series) -> str:
    """
    Original text format with simple [SEP] separators.
    
    Format: facts [SEP] applicant_reason [SEP] defence_reason
    
    This is what was used in the original training that achieved 90.7% on ALL THREE.
    """
    parts = []
    
    if 'facts' in row and pd.notna(row['facts']) and str(row['facts']).strip():
        parts.append(str(row['facts']).strip())
    
    if 'applicant_reason' in row and pd.notna(row['applicant_reason']) and str(row['applicant_reason']).strip():
        parts.append(str(row['applicant_reason']).strip())
    
    if 'defence_reason' in row and pd.notna(row['defence_reason']) and str(row['defence_reason']).strip():
        parts.append(str(row['defence_reason']).strip())
    
    return " [SEP] ".join(parts)


# ===============================================================================
# DATASET
# ===============================================================================

class LegalDataset(Dataset):
    """Dataset for Legal-BERT training."""
    
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

class MetricsLogger(TrainerCallback):
    """Log detailed metrics at each epoch."""
    
    def __init__(self):
        self.history = []
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        record = {
            'epoch': state.epoch,
            'step': state.global_step,
            **{k: v for k, v in metrics.items()}
        }
        self.history.append(record)


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
        'precision_macro': precision_score(labels, predictions, average='macro'),
        'precision_refused': precision_score(labels, predictions, pos_label=1),
        'recall_macro': recall_score(labels, predictions, average='macro'),
        'recall_refused': recall_score(labels, predictions, pos_label=1),
    }


# ===============================================================================
# DATA LOADING
# ===============================================================================

def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """Load and prepare data with ORIGINAL text format."""
    
    print(f"\n📂 Loading data from: {config.input_dataframe}")
    
    df = pd.read_pickle(config.input_dataframe)
    print(f"   Total rows: {len(df)}")
    
    # Filter to binary outcomes
    outcomes = ["summary judgment granted", "summary judgment refused"]
    df = df[df["outcome"].isin(outcomes)].copy()
    print(f"   After filtering to granted/refused: {len(df)}")
    
    # Create label
    df["label"] = df["outcome"].map({
        "summary judgment granted": 0,
        "summary judgment refused": 1,
    })
    
    # Create ORIGINAL text format
    print(f"\n   Creating text with ORIGINAL format: facts [SEP] applicant [SEP] defence")
    df["text"] = df.apply(format_text_original, axis=1)
    
    # Show example
    print(f"\n   Example text (first 200 chars):")
    print(f"   {df['text'].iloc[0][:200]}...")
    
    # Print class distribution
    print(f"\n   Class distribution:")
    n_granted = (df['label']==0).sum()
    n_refused = (df['label']==1).sum()
    print(f"   - Granted (0): {n_granted} ({n_granted/len(df)*100:.1f}%)")
    print(f"   - Refused (1): {n_refused} ({n_refused/len(df)*100:.1f}%)")
    
    # CRITICAL: Calculate class weights from ORIGINAL distribution BEFORE upsampling
    if config.calculate_weights_before_upsampling:
        total = len(df)
        weight_granted = total / (2 * n_granted)
        weight_refused = total / (2 * n_refused)
        class_weights = torch.tensor([weight_granted, weight_refused], dtype=torch.float32)
        print(f"\n   Class weights (from ORIGINAL distribution):")
        print(f"   - Granted: {weight_granted:.3f}")
        print(f"   - Refused: {weight_refused:.3f}")
    else:
        class_weights = None
    
    # Split
    train_df, val_df = train_test_split(
        df, 
        test_size=config.val_split,
        stratify=df["label"],
        random_state=config.random_seed
    )
    
    print(f"\n   Train: {len(train_df)}, Validation: {len(val_df)}")
    
    return train_df, val_df, class_weights


def upsample_minority(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """Upsample minority class to match majority."""
    
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
# TRAINING
# ===============================================================================

def train_model(config: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, class_weights: torch.Tensor) -> Tuple[Trainer, Dict, List]:
    """Train Legal-BERT with original configuration."""
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING: Legal-BERT (ORIGINAL CONFIG)")
    print(f"   Model: {config.model_name}")
    print(f"   Text format: facts [SEP] applicant [SEP] defence")
    print(f"   Class weights from ORIGINAL distribution (before upsampling)")
    print(f"{'='*70}")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Upsample
    if config.use_upsampling:
        print(f"\n   Upsampling minority class...")
        print(f"   Before: Granted={len(train_df[train_df['label']==0])}, Refused={len(train_df[train_df['label']==1])}")
        train_df = upsample_minority(train_df, config.random_seed)
        print(f"   After:  Granted={len(train_df[train_df['label']==0])}, Refused={len(train_df[train_df['label']==1])}")
        print(f"   (Class weights still use ORIGINAL distribution)")
    
    # Load tokenizer and model
    print(f"\n   Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2
    )
    
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
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
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
        logging_dir=str(output_dir / "logs"),
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Callbacks
    metrics_logger = MetricsLogger()
    early_stopping = EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
    
    # Trainer with class weights
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights if config.use_class_weights else None,
        label_smoothing=config.label_smoothing,
        callbacks=[metrics_logger, early_stopping],
    )
    
    # Train
    print(f"\n   Starting training...")
    start_time = datetime.now()
    trainer.train()
    train_time = datetime.now() - start_time
    print(f"\n   Training time: {train_time}")
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Save model
    model_path = output_dir / "best_model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"\n   ✅ Model saved to: {model_path}")
    
    return trainer, eval_results, metrics_logger.history


# ===============================================================================
# PREDICTIONS & ANALYSIS
# ===============================================================================

def generate_predictions(trainer: Trainer, val_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Generate predictions and save."""
    
    print(f"\n{'='*70}")
    print("📊 GENERATING PREDICTIONS")
    print(f"{'='*70}")
    
    predictions = trainer.predict(trainer.eval_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    
    results_df = val_df.copy().reset_index(drop=True)
    results_df['predicted'] = preds
    results_df['prob_granted'] = probs[:, 0]
    results_df['prob_refused'] = probs[:, 1]
    results_df['confidence'] = np.max(probs, axis=1)
    results_df['correct'] = results_df['label'] == results_df['predicted']
    
    def get_error_type(row):
        if row['correct']:
            return 'correct'
        elif row['predicted'] == 0:
            return 'false_positive'
        else:
            return 'false_negative'
    
    results_df['error_type'] = results_df.apply(get_error_type, axis=1)
    
    # Save
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    results_df.to_csv(output_dir / "predictions_full.csv", index=False)
    
    errors_df = results_df[~results_df['correct']]
    errors_df.to_csv(output_dir / "errors_only.csv", index=False)
    
    print(f"   ✅ Saved: predictions.csv")
    print(f"   ✅ Saved: predictions_full.csv")
    print(f"   ✅ Saved: errors_only.csv ({len(errors_df)} errors)")
    
    return results_df


def analyze_by_let_strata(results_df: pd.DataFrame, output_dir: Path):
    """Analyze performance by LAW/EVIDENCE/TRIAL strata (Table 6)."""
    
    print(f"\n{'='*70}")
    print("📊 ANALYSIS BY L/E/T STRATA (Table 6)")
    print(f"{'='*70}")
    
    cat_col = None
    for col in ['decision_reason_categories_clean', 'decision_reason_categories']:
        if col in results_df.columns:
            cat_col = col
            break
    
    if cat_col is None:
        print("   ⚠️ No L/E/T category column found")
        return
    
    def parse_categories(cat_str):
        if pd.isna(cat_str):
            return {'LAW': None, 'EVIDENCE': None, 'TRIAL': None}
        cats = {}
        for item in str(cat_str).split():
            if '=' in item:
                key, val = item.split('=')
                cats[key] = int(val) if val.isdigit() else None
        return cats
    
    results_df['_cats'] = results_df[cat_col].apply(parse_categories)
    results_df['LAW'] = results_df['_cats'].apply(lambda x: x.get('LAW'))
    results_df['EVIDENCE'] = results_df['_cats'].apply(lambda x: x.get('EVIDENCE'))
    results_df['TRIAL'] = results_df['_cats'].apply(lambda x: x.get('TRIAL'))
    
    strata = [
        ('ALL THREE', (results_df['LAW'] == 1) & (results_df['EVIDENCE'] == 1) & (results_df['TRIAL'] == 1)),
        ('LAW + TRIAL', (results_df['LAW'] == 1) & (results_df['EVIDENCE'] == 0) & (results_df['TRIAL'] == 1)),
        ('LAW + EVIDENCE', (results_df['LAW'] == 1) & (results_df['EVIDENCE'] == 1) & (results_df['TRIAL'] == 0)),
        ('EVIDENCE + TRIAL', (results_df['LAW'] == 0) & (results_df['EVIDENCE'] == 1) & (results_df['TRIAL'] == 1)),
        ('LAW only', (results_df['LAW'] == 1) & (results_df['EVIDENCE'] == 0) & (results_df['TRIAL'] == 0)),
        ('EVIDENCE only', (results_df['LAW'] == 0) & (results_df['EVIDENCE'] == 1) & (results_df['TRIAL'] == 0)),
        ('TRIAL only', (results_df['LAW'] == 0) & (results_df['EVIDENCE'] == 0) & (results_df['TRIAL'] == 1)),
    ]
    
    print(f"\n   {'Category':<20} {'N':>6} {'Acc':>8} {'F1-Mac':>8} {'F1-Gr':>8} {'F1-Ref':>8}")
    print(f"   {'-'*58}")
    
    stratum_results = []
    
    for name, mask in strata:
        subset = results_df[mask]
        if len(subset) < 2:
            continue
        
        y_true = subset['label'].values
        y_pred = subset['predicted'].values
        
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro')
        f1_gr = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_ref = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        print(f"   {name:<20} {len(subset):>6} {acc:>7.1%} {f1_mac:>8.3f} {f1_gr:>8.3f} {f1_ref:>8.3f}")
        
        stratum_results.append({
            'category': name,
            'n': len(subset),
            'accuracy': acc,
            'f1_macro': f1_mac,
            'f1_granted': f1_gr,
            'f1_refused': f1_ref,
        })
    
    # Overall
    y_true = results_df['label'].values
    y_pred = results_df['predicted'].values
    print(f"   {'-'*58}")
    print(f"   {'OVERALL':<20} {len(results_df):>6} {accuracy_score(y_true, y_pred):>7.1%} "
          f"{f1_score(y_true, y_pred, average='macro'):>8.3f} "
          f"{f1_score(y_true, y_pred, pos_label=0):>8.3f} "
          f"{f1_score(y_true, y_pred, pos_label=1):>8.3f}")
    
    pd.DataFrame(stratum_results).to_csv(output_dir / "let_stratum_analysis.csv", index=False)
    print(f"\n   ✅ Saved: let_stratum_analysis.csv")


def plot_learning_curves(history: List[Dict], output_dir: Path):
    """Plot learning curves."""
    
    print(f"\n{'='*70}")
    print("📈 PLOTTING LEARNING CURVES")
    print(f"{'='*70}")
    
    if not history:
        print("   ⚠️ No training history available")
        return
    
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss
    if 'eval_loss' in df.columns:
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['eval_loss'], 'b-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.grid(True, alpha=0.3)
    
    # F1 Macro
    if 'eval_f1_macro' in df.columns:
        ax = axes[0, 1]
        ax.plot(df['epoch'], df['eval_f1_macro'], 'g-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Macro')
        ax.set_title('F1-Macro Score')
        ax.grid(True, alpha=0.3)
        
        best_idx = df['eval_f1_macro'].idxmax()
        best_epoch = df.loc[best_idx, 'epoch']
        best_f1 = df.loc[best_idx, 'eval_f1_macro']
        ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {best_f1:.3f}', xy=(best_epoch, best_f1), 
                   xytext=(best_epoch+0.5, best_f1-0.02))
    
    # F1 by class
    ax = axes[0, 2]
    if 'eval_f1_granted' in df.columns:
        ax.plot(df['epoch'], df['eval_f1_granted'], 'b-o', label='F1-Granted')
    if 'eval_f1_refused' in df.columns:
        ax.plot(df['epoch'], df['eval_f1_refused'], 'r-o', label='F1-Refused')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    if 'eval_accuracy' in df.columns:
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['eval_accuracy'], 'm-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.grid(True, alpha=0.3)
    
    # Precision
    if 'eval_precision_macro' in df.columns:
        ax = axes[1, 1]
        ax.plot(df['epoch'], df['eval_precision_macro'], 'c-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision (Macro)')
        ax.set_title('Precision')
        ax.grid(True, alpha=0.3)
    
    # Recall
    if 'eval_recall_macro' in df.columns:
        ax = axes[1, 2]
        ax.plot(df['epoch'], df['eval_recall_macro'], 'y-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall (Macro)')
        ax.set_title('Recall')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Legal-BERT Training Progress (ORIGINAL CONFIG)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: learning_curves.png")
    
    df.to_csv(output_dir / 'training_history.csv', index=False)
    print(f"   ✅ Saved: training_history.csv")


def plot_confusion_matrix(results_df: pd.DataFrame, output_dir: Path):
    """Plot confusion matrix."""
    
    y_true = results_df['label'].values
    y_pred = results_df['predicted'].values
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Granted', 'Refused'],
                yticklabels=['Granted', 'Refused'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Legal-BERT (ORIGINAL CONFIG)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: confusion_matrix.png")


def save_summary(eval_results: Dict, config: Config, output_dir: Path):
    """Save summary."""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': config.model_name,
        'text_format': 'ORIGINAL: facts [SEP] applicant [SEP] defence',
        'class_weights_source': 'ORIGINAL distribution (before upsampling)',
        'config': asdict(config),
        'results': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in eval_results.items()},
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   ✅ Saved: summary.json")


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("🚀 LEGAL-BERT TRAINING - ORIGINAL CONFIGURATION")
    print("="*70)
    print("   Replicating the exact config that achieved 90.7% on ALL THREE")
    print("")
    print("   Key settings:")
    print("   - Text: facts [SEP] applicant [SEP] defence")
    print("   - Class weights from ORIGINAL distribution (before upsampling)")
    print("="*70)
    print(f"\n   Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration
    config = Config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data file
    if not Path(config.input_dataframe).exists():
        print(f"\n   ❌ ERROR: Data file not found: {config.input_dataframe}")
        print(f"   Please place {config.input_dataframe} in the current directory")
        return
    
    # Load data (with class weights calculated from ORIGINAL distribution)
    train_df, val_df, class_weights = load_data(config)
    
    # Train
    trainer, eval_results, history = train_model(config, train_df, val_df, class_weights)
    
    # Generate predictions
    results_df = generate_predictions(trainer, val_df, output_dir)
    
    # Analysis
    analyze_by_let_strata(results_df, output_dir)
    plot_learning_curves(history, output_dir)
    plot_confusion_matrix(results_df, output_dir)
    save_summary(eval_results, config, output_dir)
    
    # Print final results
    print("\n" + "="*70)
    print("📋 FINAL RESULTS (ORIGINAL CONFIG)")
    print("="*70)
    print(f"\n   Accuracy:         {eval_results.get('eval_accuracy', 0):.1%}")
    print(f"   Balanced Acc:     {eval_results.get('eval_balanced_accuracy', 0):.1%}")
    print(f"   F1-Macro:         {eval_results.get('eval_f1_macro', 0):.3f}")
    print(f"   F1-Granted:       {eval_results.get('eval_f1_granted', 0):.3f}")
    print(f"   F1-Refused:       {eval_results.get('eval_f1_refused', 0):.3f}")
    
    print(f"\n   📁 All outputs saved to: {output_dir}/")
    print(f"\n   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
