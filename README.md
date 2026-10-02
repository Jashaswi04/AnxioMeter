# AnxioMeter
- Purpose: synthetic prototype demonstrating ingestion, normalization, topic modeling, and Phase 2 handoff.
- Local execution order:
.\.venv\Scripts\python.exe dev1_pipeline\00_expand_sample_sources.py
.\.venv\Scripts\python.exe dev1_pipeline\01_scrape_reddit.py
.\.venv\Scripts\python.exe dev1_pipeline\02_scrape_linkedin.py
.\.venv\Scripts\python.exe dev1_pipeline\03_lexicon_cleaner.py

- Modeling: use the saved Colab notebook; upload the newly cleaned corpus.
- Current dataset: 360 synthetic records, 180 per platform.
- Current model: three assigned topics, 108 outliers.
- Main limitation: one broad mixed topic; results do not establish real platform differences.
- Phase 2 entry point: data/topic_mapped_data.csv.
- Text column for emotion and fluff scoring: text_for_analysis.
- Rebuild warning: regenerating samples replaces demo inputs; review newly trained topic IDs before labeling them.