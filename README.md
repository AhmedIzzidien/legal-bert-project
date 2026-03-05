# LegalBERT Summary Judgement Classification

Replication code for analyzing domain-modulated outcome bias in LegalBERT classification of UK judicial review cases.

## Key Finding

The model exhibits **domain-modulated outcome bias**: error rates for refused cases reach 32% in specific legal domains (e.g., contract, defamation) compared to just 5% for granted cases in the same domains. The model has learned spurious associations between domain-specific legal vocabulary and the "granted" outcome.

## Scripts

| Script | Function |
|--------|----------|
| `legalbert_multiseed_cv_attention.py` | **Training**: 5-seed × 5-fold cross-validation with attention extraction |
| `extract_multiseed_results.py` | **Results aggregation**: consistency scores (0/5 to 5/5), per-seed metrics, FP/FN breakdown |
| `compute_stratum_f1_multiseed.py` | **L/E/T analysis**: F1 scores by Law/Evidence/Trial stratum combinations |
| `attention_ablation_analysis.py` | **Causal analysis**: attention density extraction, section ablation experiments |
| `multiseed_phrase_analysis_final.py` | **Phrase analysis**: n-gram extraction, Fisher's exact tests, Benjamini-Hochberg FDR correction |
| `multiseed_topic_analysis_final.py` | **Topic analysis**: topic × outcome error rates, cross-topic consistency |
| `statistical_analysis.py` | **Statistical tests**: chi-square, Fisher's exact, logistic regression, Wilson CIs, calibration (ECE) |
| `check_keyword_outcome.py` | **Interaction analysis**: keyword × outcome error rate stratification |
| `verify_paper_final.py` | **Verification**: confirms all reported statistics match raw data |

## Data

* `sj_231025.pkl` - Main dataset (N=2,109 cases, 1,961 binary outcomes)
* `sj_231025_w_topics_all_cases.pkl` - Dataset with topic assignments

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

# Analysis pipeline
python extract_multiseed_results.py
python compute_stratum_f1_multiseed.py
python attention_ablation_analysis.py
python multiseed_topic_analysis_final.py
python multiseed_phrase_analysis_final.py
python statistical_analysis.py
python check_keyword_outcome.py

# Verify all paper statistics
python verify_paper_final.py
```

## Output Files

The scripts generate:
- `legalbert_multiseed_attention/` - Model checkpoints and predictions
- `attention_fixed_v2/` - Attention weights and ablation results
- `multiseed_topic_analysis/` - Topic error rate tables
- `multiseed_phrase_analysis/` - Phrase differential analysis

## Citation

If you use this code, please cite: [paper reference]
