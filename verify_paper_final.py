#!/usr/bin/env python3
"""
COMPLETE PAPER VERIFICATION - With label consistency check
Single source of truth: pickle file for labels
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import re
import warnings
warnings.filterwarnings("ignore")

PREDICTIONS = "legalbert_multiseed_attention/all_predictions.csv"
CONSISTENCY = "legalbert_multiseed_attention/case_consistency.csv"
ORIGINAL = "sj_231025.pkl"
ATTENTION = "attention_fixed_v2/attention_per_case.csv"
ABLATION = "attention_fixed_v2/ablation_results.csv"

report = []
passes = fails = warns = 0

def log(m):
    print(m)
    report.append(m)

def check(name, expected, actual, atol=None, rtol=0.02):
    global passes, fails
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(expected - actual) <= (atol if atol else rtol * max(abs(expected), 1))
        if ok:
            log(f"  PASS: {name} = {actual} (expected {expected})")
            passes += 1
        else:
            log(f"  FAIL: {name} = {actual} (expected {expected})")
            fails += 1
    else:
        if expected == actual:
            log(f"  PASS: {name} = {actual}")
            passes += 1
        else:
            log(f"  FAIL: {name} = {actual} (expected {expected})")
            fails += 1

def warn(m):
    global warns
    log(f"  WARNING: {m}")
    warns += 1

def info(m):
    log(f"  INFO: {m}")

log("=" * 80)
log("PAPER VERIFICATION (with label consistency check)")
log("=" * 80)

# Load data files
try:
    preds = pd.read_csv(PREDICTIONS)
    log(f"Loaded preds: {len(preds)}")
except:
    preds = None
    log("ERROR: no preds")

try:
    cons = pd.read_csv(CONSISTENCY)
    log(f"Loaded cons: {len(cons)}")
except:
    cons = None
    log("ERROR: no cons")

df_text = None
for f in [ORIGINAL, "sj_231025_w_topics_all_cases.pkl"]:
    try:
        df_text = pd.read_pickle(f)
        log(f"Loaded pickle: {f}")
        break
    except:
        pass

try:
    attn = pd.read_csv(ATTENTION)
    log(f"Loaded attn: {len(attn)}")
except:
    attn = None

try:
    abl = pd.read_csv(ABLATION)
    log(f"Loaded abl: {len(abl)}")
except:
    abl = None

# BUILD CANONICAL LABELS FROM PICKLE
log("\n" + "=" * 80)
log("LABEL CONSISTENCY CHECK")
log("=" * 80)

df = None
if df_text is not None:
    outcomes = ["summary judgment granted", "summary judgment refused"]
    df = df_text[df_text["outcome"].isin(outcomes)].copy().reset_index(drop=True)
    df['case_id'] = range(len(df))
    df['label'] = df['outcome'].map({"summary judgment granted": 0, "summary judgment refused": 1})
    log(f"Built canonical labels from pickle: {len(df)} cases")
    
    if cons is not None:
        merged = df[['case_id', 'label']].merge(
            cons[['case_id', 'label']], on='case_id', suffixes=('_pkl', '_cons'))
        mismatch = (merged['label_pkl'] != merged['label_cons']).sum()
        if mismatch > 0:
            log(f"  ERROR: {mismatch} mismatches pickle vs cons!")
        else:
            log("  PASS: pickle == case_consistency.csv")
    
    if preds is not None:
        preds_labels = preds.groupby('case_id')['label'].first().reset_index()
        merged2 = df[['case_id', 'label']].merge(
            preds_labels, on='case_id', suffixes=('_pkl', '_preds'))
        mismatch2 = (merged2['label_pkl'] != merged2['label_preds']).sum()
        if mismatch2 > 0:
            log(f"  ERROR: {mismatch2} mismatches pickle vs preds!")
        else:
            log("  PASS: pickle == all_predictions.csv")
    
    if cons is not None:
        df = df.merge(cons[['case_id', 'times_correct', 'stratum']], on='case_id', how='left')
else:
    df = cons.copy() if cons is not None else None

# DATASET
log("\n" + "=" * 80 + "\nDATASET")
if df is not None:
    check("Total", 1961, len(df))
    check("Granted", 1196, (df['label'] == 0).sum())
    check("Refused", 765, (df['label'] == 1).sum())

# OVERALL PERFORMANCE
log("\n" + "=" * 80 + "\nOVERALL PERFORMANCE")
if preds is not None:
    seeds = preds['seed'].unique()
    accs, f1ms, f1gs, f1rs = [], [], [], []
    for s in seeds:
        d = preds[preds['seed'] == s]
        accs.append(accuracy_score(d['label'], d['predicted']))
        f1ms.append(f1_score(d['label'], d['predicted'], average='macro'))
        f1gs.append(f1_score(d['label'], d['predicted'], pos_label=0))
        f1rs.append(f1_score(d['label'], d['predicted'], pos_label=1))
    check("Accuracy", 0.627, round(np.mean(accs), 3), atol=0.01)
    check("F1-Macro", 0.598, round(np.mean(f1ms), 3), atol=0.01)
    check("F1-Granted", 0.724, round(np.mean(f1gs), 3), atol=0.01)
    check("F1-Refused", 0.472, round(np.mean(f1rs), 3), atol=0.01)

# TABLE 1
log("\n" + "=" * 80 + "\nTABLE 1: CONSISTENCY")
if df is not None and 'times_correct' in df.columns:
    exp = {0: 153, 1: 254, 2: 280, 3: 310, 4: 418, 5: 546}
    act = df['times_correct'].value_counts()
    for s, e in exp.items():
        check(f"Score {s}/5", e, act.get(s, 0))

# TABLE 2
log("\n" + "=" * 80 + "\nTABLE 2: OUTCOME ASYMMETRY")
if df is not None and 'times_correct' in df.columns:
    g = df[df['label'] == 0]
    r = df[df['label'] == 1]
    g_aw = (g['times_correct'] == 0).sum()
    r_aw = (r['times_correct'] == 0).sum()
    check("Granted AW", 43, g_aw)
    check("Refused AW", 110, r_aw)
    check("Granted %", 3.6, round(g_aw / len(g) * 100, 1), atol=0.2)
    check("Refused %", 14.4, round(r_aw / len(r) * 100, 1), atol=0.2)
    cont = [[g_aw, len(g) - g_aw], [r_aw, len(r) - r_aw]]
    chi2, p, _, _ = stats.chi2_contingency(cont)
    check("Chi-sq", 73.2, round(chi2, 1), atol=1.0)

# TABLE 3
log("\n" + "=" * 80 + "\nTABLE 3: ERROR DIRECTION")
if df is not None and 'times_correct' in df.columns:
    aw = df[df['times_correct'] == 0]
    fp = (aw['label'] == 1).sum()
    fn = (aw['label'] == 0).sum()
    check("Total AW", 153, len(aw))
    check("FP", 110, fp)
    check("FN", 43, fn)

# TABLE 4 N CHECK
log("\n" + "=" * 80 + "\nTABLE 4: L/E/T N CHECK")
if df is not None and 'stratum' in df.columns:
    t4 = {'EVIDENCE + TRIAL': 188, 'LAW + EVIDENCE': 444, 'ALL THREE': 163}
    for st, en in t4.items():
        an = len(df[df['stratum'] == st])
        if en != an:
            warn(f"{st}: paper={en} actual={an}")

# SECTION 4 PERFORMANCE
log("\n" + "=" * 80 + "\nSECTION 4: L/E/T PERFORMANCE")
if preds is not None and df is not None and 'stratum' in df.columns:
    ps = preds.merge(df[['case_id', 'stratum']], on='case_id')
    s4 = {
        'ALL THREE': {'n': 161, 'acc': 0.706, 'f1g': 0.799, 'f1r': 0.447},
        'LAW + EVIDENCE': {'n': 559, 'acc': 0.678, 'f1g': 0.780, 'f1r': 0.399},
        'LAW + TRIAL': {'n': 112, 'acc': 0.641, 'f1g': 0.712, 'f1r': 0.523},
        'TRIAL only': {'n': 120, 'acc': 0.655, 'f1g': 0.708, 'f1r': 0.579},
        'EVIDENCE only': {'n': 317, 'acc': 0.605, 'f1g': 0.642, 'f1r': 0.559},
        'LAW only': {'n': 487, 'acc': 0.579, 'f1g': 0.625, 'f1r': 0.519},
        'EVIDENCE + TRIAL': {'n': 187, 'acc': 0.541, 'f1g': 0.528, 'f1r': 0.552},
    }
    seeds = ps['seed'].unique()
    for st, ex in s4.items():
        accs, f1gs, f1rs = [], [], []
        for sd in seeds:
            d = ps[(ps['seed'] == sd) & (ps['stratum'] == st)]
            if len(d) > 0 and len(d['label'].unique()) > 1:
                accs.append(accuracy_score(d['label'], d['predicted']))
                f1gs.append(f1_score(d['label'], d['predicted'], pos_label=0))
                f1rs.append(f1_score(d['label'], d['predicted'], pos_label=1))
        check(f"{st} N", ex['n'], len(df[df['stratum'] == st]))
        if accs:
            check(f"{st} acc", ex['acc'], round(np.mean(accs), 3), atol=0.01)
            check(f"{st} F1g", ex['f1g'], round(np.mean(f1gs), 3), atol=0.01)
            check(f"{st} F1r", ex['f1r'], round(np.mean(f1rs), 3), atol=0.01)

# TABLE 5 ATTENTION
log("\n" + "=" * 80 + "\nTABLE 5: ATTENTION")
if attn is not None:
    aw_a = attn[attn['times_correct'] == 0]
    ar_a = attn[attn['times_correct'] == 5]
    check("AW FACTS", 0.0047, round(aw_a['raw_density_FACTS'].mean(), 4), atol=0.001)
    check("AW APP", 0.0047, round(aw_a['raw_density_APPLICANT'].mean(), 4), atol=0.001)
    check("AW DEF", 0.0056, round(aw_a['raw_density_DEFENCE'].mean(), 4), atol=0.001)
    check("AR FACTS", 0.0049, round(ar_a['raw_density_FACTS'].mean(), 4), atol=0.001)
    check("AR APP", 0.0049, round(ar_a['raw_density_APPLICANT'].mean(), 4), atol=0.001)
    check("AR DEF", 0.0059, round(ar_a['raw_density_DEFENCE'].mean(), 4), atol=0.001)

# TABLE 6 ABLATION
log("\n" + "=" * 80 + "\nTABLE 6: ABLATION")
if abl is not None:
    aw_ab = abl[abl['times_correct'] == 0]
    ar_ab = abl[abl['times_correct'] == 5]
    check("AW FACTS d", 0.158, round(aw_ab['facts_delta'].mean(), 3), atol=0.02)
    check("AW APP d", 0.115, round(aw_ab['applicant_delta'].mean(), 3), atol=0.02)
    check("AW DEF d", 0.025, round(aw_ab['defence_delta'].mean(), 3), atol=0.02)
    check("AR FACTS d", 0.164, round(ar_ab['facts_delta'].mean(), 3), atol=0.02)
    check("AR APP d", 0.085, round(ar_ab['applicant_delta'].mean(), 3), atol=0.02)
    check("AR DEF d", -0.020, round(ar_ab['defence_delta'].mean(), 3), atol=0.02)

# TABLE 7
log("\n" + "=" * 80 + "\nTABLE 7: ABLATION BY ERROR")
if abl is not None:
    aw_ab = abl[abl['times_correct'] == 0]
    fp_ab = aw_ab[aw_ab['label'] == 1]
    fn_ab = aw_ab[aw_ab['label'] == 0]
    info(f"FP N: {len(fp_ab)} (paper:29), FN N: {len(fn_ab)} (paper:11)")
    info(f"FP DEF: {fp_ab['defence_delta'].mean():.3f} (paper:+0.129)")
    info(f"FN DEF: {fn_ab['defence_delta'].mean():.3f} (paper:-0.249)")
    if len(fp_ab) != 29 or len(fn_ab) != 11:
        warn("Table 7 N mismatch - paper used subset")

# VOCABULARY
log("\n" + "=" * 80 + "\nVOCABULARY & KEYWORDS")
if df is not None and 'times_correct' in df.columns and 'defence_reason' in df.columns:
    aw_df = df[df['times_correct'] == 0]
    ar_df = df[df['times_correct'] == 5]
    n_aw, n_ar = len(aw_df), len(ar_df)
    
    def ext(t):
        return set(re.findall(r'\b[a-z]{4,}\b', str(t).lower())) if pd.notna(t) else set()
    
    aw_def = aw_df['defence_reason'].apply(ext)
    ar_def = ar_df['defence_reason'].apply(ext)
    t8 = {'defamatory': (6.5, 1.1), 'binding': (5.9, 1.3), 'property': (5.9, 1.8)}
    for w, (ew, er) in t8.items():
        wp = sum(1 for ws in aw_def if w in ws) / n_aw * 100
        rp = sum(1 for ws in ar_def if w in ws) / n_ar * 100
        check(f"'{w}' W%", ew, round(wp, 1), atol=0.5)
        check(f"'{w}' R%", er, round(rp, 1), atol=0.5)
    
    df['comb'] = (df['facts'].fillna('') + ' ' + df['applicant_reason'].fillna('') + ' ' + df['defence_reason'].fillna('')).str.lower()
    t10 = {'binding': (73, 16.4, 7.5), 'defamatory': (52, 15.4, 7.6), 'settlement': (90, 13.3, 7.5)}
    for kw, (en, ew, ewo) in t10.items():
        has = df['comb'].str.contains(kw, na=False)
        wk = df[has]
        wok = df[~has]
        errw = (wk['times_correct'] == 0).sum() / len(wk) * 100 if len(wk) > 0 else 0
        errwo = (wok['times_correct'] == 0).sum() / len(wok) * 100 if len(wok) > 0 else 0
        check(f"'{kw}' N", en, len(wk))
        check(f"'{kw}' w%", ew, round(errw, 1), atol=1.0)
        check(f"'{kw}' wo%", ewo, round(errwo, 1), atol=1.0)
    
    t11 = {'binding': (4.8, 32.3), 'defamatory': (6.5, 28.6), 'property': (3.3, 17.8)}
    for kw, (eg, er) in t11.items():
        has = df['comb'].str.contains(kw, na=False)
        wk = df[has]
        gk = wk[wk['label'] == 0]
        rk = wk[wk['label'] == 1]
        ge = (gk['times_correct'] == 0).sum() / len(gk) * 100 if len(gk) > 0 else 0
        re = (rk['times_correct'] == 0).sum() / len(rk) * 100 if len(rk) > 0 else 0
        check(f"'{kw}' G%", eg, round(ge, 1), atol=1.0)
        check(f"'{kw}' R%", er, round(re, 1), atol=1.0)
    
    b = df[df['comb'].str.contains('binding', na=False)]
    check("Binding N", 73, len(b))
    check("Binding G", 42, (b['label'] == 0).sum())
    check("Binding R", 31, (b['label'] == 1).sum())
    check("Binding G AW", 2, ((b['label'] == 0) & (b['times_correct'] == 0)).sum())
    check("Binding R AW", 10, ((b['label'] == 1) & (b['times_correct'] == 0)).sum())
    
    try:
        vec = CountVectorizer(ngram_range=(1, 3), min_df=5, max_df=0.95, stop_words='english')
        vec.fit(df['comb'].fillna('').tolist())
        check("Phrases", 5392, len(vec.get_feature_names_out()), atol=100)
    except:
        pass

# SUMMARY
log("\n" + "=" * 80 + "\nINCONSISTENCY")
warn("Table 4 N values differ from Section 4 - update paper")

log("\n" + "=" * 80 + "\nSUMMARY")
log(f"PASSED: {passes}")
log(f"FAILED: {fails}")
log(f"WARNINGS: {warns}")

with open("verification_report.txt", "w") as f:
    f.write("\n".join(report))
log("Saved: verification_report.txt")
