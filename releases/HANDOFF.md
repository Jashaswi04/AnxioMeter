# AnxioMeter Phase 1 — Synthetic Prototype

## Start here
Load data/topic_mapped_data.csv with pandas.
No scraping or topic retraining is required to begin Phase 2.

## Column usage
- id: stable record identifier; preserve it in downstream outputs.
- platform: synthetic platform assignment, reddit or linkedin.
- created_utc: invented publication timestamp.
- text: original synthetic wording.
- text_for_analysis: input for emotion classification and fluff scoring.
- cleaned_text: normalized input used for topic modeling.
- topic_id: assigned topic; -1 means Outlier / Unassigned.
- topic_name: provisional human-reviewed topic label.
- is_synthetic: must remain attached to downstream records.
- model_version: identifies this dataset/model release.

## Analysis rules
Topic 0 is a mixed workplace-concerns cluster.
Do not rename it to a narrower theme without reviewing its records.
Report outliers separately or state explicitly when excluding them.
Never present these synthetic platform comparisons as research findings.
The samples do not establish Reddit/LinkedIn sentiment differences.

## Model
models/bertopic_model contains a compact safetensors BERTopic model.
The embedding model may download on first use.
Compact inference uses topic embeddings and can produce assignments
different from the original density-based training assignments.
Use the topic IDs already stored in the CSV for this corpus.

## Reproducibility
reports/run_metadata.json records settings and package versions.
requirements-modeling.txt pins the listed modeling packages from Colab;
it is not a complete lock of all transitive dependencies.
Keep the completed notebook and source-generation/cleaning scripts
alongside this release. The notebook captures keyword-extraction settings.

## Phase 2
Add emotion and index columns while preserving record IDs and provenance.
Write the resulting dataset to data/final_analytics.csv.
