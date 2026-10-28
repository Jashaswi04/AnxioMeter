# Developer 2: run Phase 2

Use Python 3.10–3.12 in a fresh virtual environment. On Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase2.txt
.\.venv\Scripts\python.exe dev2_pipeline\05_emotion_classifier.py
.\.venv\Scripts\python.exe dev2_pipeline\06_fluff_calculator.py
.\.venv\Scripts\python.exe dev2_pipeline\07_divergence_matrix.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `py -3.12` is unavailable, install Python 3.12 or use a compatible Python 3.10/3.11 interpreter. The first scoring run downloads the GoEmotions model and can take time. Do not retrain BERTopic for Phase 2: the CSV already has topic IDs. The input text is `text_for_analysis`; original IDs, timestamps, and provenance are retained.

## Outputs

- `data/emotion_scored.csv`: original columns plus independent model probabilities for fear, nervousness, anger, annoyance, admiration, gratitude, optimism, and sadness.
- `data/final_analytics.csv`: emotion scores plus CAI, CFS, buzzword count, announcement phrase count, and type-token ratio.
- `data/divergence_summary.csv`: record counts and mean indices by topic and platform, including topic `-1` unless filtered in the app.

## Score definitions

Scores are **illustrative indices on a 0–100 scale**, not validated measures of psychological state or proof of corporate intent.

`CAI = 100 × (0.30 fear + 0.25 nervousness + 0.20 anger + 0.15 annoyance + 0.10 sadness)`

`CFS = 100 × (0.45 buzzword_density + 0.40 announcement_density + 0.15 (1 − type_token_ratio))`

Here `buzzword_density = min(1, 10 × buzzword_count / word_count)` and `announcement_density = min(1, 20 × announcement_count / word_count)`; an empty post scores zero. The phrase counts are case insensitive. Because type-token ratio depends strongly on text length, compare CFS cautiously across posts of different lengths. Record counts appear in the dashboard for every group.

The source dataset is **synthetic**: the graphs demonstrate a working comparison workflow, but they cannot establish differences in actual Reddit and LinkedIn users. Topic 0 is a broad mixed cluster; topic -1 is unassigned. The translator is a deterministic, humorous paraphrase and should not be presented as mind reading or as a learned Reddit style model.
