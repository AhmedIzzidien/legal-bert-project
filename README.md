# LegalBERT Judicial Review Classification

Replication code for analyzing domain-modulated outcome bias in LegalBERT classification of UK judicial review cases.

## Scripts

| Script | Description |
|--------|-------------|
| `legalbert_multiseed_cv_attention.py` | Main training: 5-seed x 5-fold cross-validation |
| `compute_stratum_f1_multiseed.py` | F1 scores by Leave/Entry/Transit strata |
| `fix_attention_v2.py` | Attention extraction and token ablation study |
| `multiseed_topic_analysis_final.py` | Topic x outcome error analysis |
| `multiseed_phrase_analysis_final.py` | Domain phrase pattern analysis |
| `check_keyword_outcome.py` | Keyword x outcome interaction effects |
| `statistical_analysis.py` | McNemar's test, bootstrap CIs |
| `extract_multiseed_results.py` | Aggregate multi-seed results |
| `verify_paper_final.py` | Verify all reported statistics |

## Data

- `sj_231025.pkl` - Main dataset (N=2,109 cases)
- `sj_231025_w_topics_all_cases.pkl` - Dataset with LDA topic assignments

## Requirements

```
torch>=2.0
transformers>=4.30
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.11
statsmodels>=0.14
```

## Usage

```bash
# Training (~8-12 hours on GPU)
python legalbert_multiseed_cv_attention.py

# Analysis
python compute_stratum_f1_multiseed.py
python fix_attention_v2.py
python multiseed_topic_analysis_final.py
python statistical_analysis.py
```
