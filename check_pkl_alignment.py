#!/usr/bin/env python3
"""
Quick check: Do sj_231025.pkl and sj_231025_w_topics_all_cases.pkl have matching rows?
"""

import pandas as pd

# Load both files
df_multiseed = pd.read_pickle("sj_231025.pkl")
df_topics = pd.read_pickle("sj_231025_w_topics_all_cases.pkl")

print("="*60)
print("COMPARING DATAFRAMES")
print("="*60)

print(f"\n📊 sj_231025.pkl (multi-seed source):")
print(f"   Rows: {len(df_multiseed)}")
print(f"   Columns: {list(df_multiseed.columns)}")

print(f"\n📊 sj_231025_w_topics_all_cases.pkl (with topics):")
print(f"   Rows: {len(df_topics)}")
print(f"   Columns: {list(df_topics.columns)}")

# Check for common ID columns
possible_ids = ['case_id', 'id', 'index', 'citation', 'case_name', 'neutral_citation']
common_ids = [col for col in possible_ids if col in df_multiseed.columns and col in df_topics.columns]

print(f"\n🔑 Potential ID columns in both: {common_ids}")

# Quick row count check
if len(df_multiseed) == len(df_topics):
    print(f"\n✅ SAME ROW COUNT: {len(df_multiseed)}")
else:
    print(f"\n⚠️  DIFFERENT ROW COUNTS: {len(df_multiseed)} vs {len(df_topics)}")

# If there's a common ID, check overlap
if common_ids:
    id_col = common_ids[0]
    ids_multi = set(df_multiseed[id_col].values)
    ids_topics = set(df_topics[id_col].values)
    
    overlap = len(ids_multi & ids_topics)
    only_multi = len(ids_multi - ids_topics)
    only_topics = len(ids_topics - ids_multi)
    
    print(f"\n🔍 Using '{id_col}' as identifier:")
    print(f"   Overlap: {overlap}")
    print(f"   Only in multi-seed: {only_multi}")
    print(f"   Only in topics: {only_topics}")
    
    if overlap == len(df_multiseed) == len(df_topics):
        print(f"\n✅ PERFECT MATCH - Can merge topics onto multi-seed results!")
    elif overlap > 0:
        print(f"\n⚠️  PARTIAL MATCH - {overlap}/{len(df_multiseed)} rows can be merged")
else:
    # Try index-based comparison
    print("\n🔍 No common ID column. Checking if indices align...")
    
    # Check first few text fields for similarity
    text_cols = [c for c in df_multiseed.columns if 'fact' in c.lower() or 'reason' in c.lower()]
    if text_cols:
        col = text_cols[0]
        if col in df_topics.columns:
            match_count = sum(df_multiseed[col].iloc[:100] == df_topics[col].iloc[:100])
            print(f"   First 100 rows of '{col}': {match_count}/100 match")

# Show topic columns available
topic_cols = [c for c in df_topics.columns if 'topic' in c.lower()]
print(f"\n📋 Topic columns available: {topic_cols}")

# Show sample of topics
if topic_cols:
    print(f"\n📋 Sample topics ({topic_cols[0]}):")
    print(df_topics[topic_cols[0]].value_counts().head(10))
