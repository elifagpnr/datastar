# DataLeague Datathon Runner

This repo contains an explainable unsupervised manipulation/anomaly scoring pipeline for `dataset/datathonFINAL.parquet`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_smoke_tests.py
```

Fast local run:

```bash
python datathon_pipeline.py run --sample-size 50000 --skip-full-author-stats
```

Final run:

```bash
python datathon_pipeline.py run --sample-size 200000
```

Live prediction:

```bash
python datathon_pipeline.py predict "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com" \
  --english-keywords "buy, free, promo, deal"
```

Batch CSV prediction for jury-provided texts:

```bash
python datathon_pipeline.py predict-csv jury_inputs/jury_texts.csv --output artifacts/jury_predictions.csv
```

CSV handling:

- If the file has an `original_text` column, it is used automatically.
- If the file has a single text column, it is used automatically.
- If the file has no header, add `--no-header`.
- If the file has multiple text cells across columns, add `--all-cells`.
- If the text column has a different name, pass `--text-column "column_name"`.
- If the CSV uses semicolon delimiters, pass `--sep ";"`.

Output includes `label`, `risk_band`, `risk_score`, `organic_score`, `top_reasons`, `nlp_text_risk`, and source row/column metadata.

Notebook workflow for the presentation:

1. Put the jury CSV file under `jury_inputs/`.
2. Open `ads.ipynb`.
3. Go to `## 8. Jury CSV Batch Inference`.
4. Change only:

```python
JURY_CSV_FILENAME = "jury_texts.csv"
```

5. Run the cell. It writes `artifacts/jury_predictions_<file_name>.csv` and displays the risk-band summary plus the first scored rows.

If your system Python says `externally-managed-environment`, use the virtualenv commands above. In VS Code/Jupyter, select `.venv/bin/python` as the notebook kernel before running `ads.ipynb`.

## Text-Only NLP Layer

The main system remains an explainable unsupervised risk scoring pipeline. For jury-provided `original_text` inputs, the pipeline also trains a lightweight pseudo-labeled NLP text model during `run`:

- positive pseudo-labels: high-risk, non-empty texts with strong manipulation reason codes
- negative pseudo-labels: low-risk, non-empty texts with `LOW_CONTENT_RISK`
- model: hashed word/character n-gram log-odds classifier implemented with `numpy`
- outputs: `artifacts/nlp_text_model.npz`, `artifacts/nlp_text_model_metadata.json`, `artifacts/nlp_pseudo_label_training_summary.csv`

`predict_live()` reports `nlp_text_risk` and `nlp_model_used`. This is not a label-based supervised accuracy claim; it is a weakly supervised helper for more stable text-only inference.

## Main Files

- `ads.ipynb`: jury-facing notebook with dataset check, scoring, visualizations, and live demo cell.
- `datathon_pipeline.py`: reusable scoring pipeline and `predict_live()` function.
- `datathon_pitch_deck.md`: static 8-slide pitch deck template.
- `artifacts/pitch_deck.md`: generated pitch deck after running the pipeline.

## Generated Artifacts

The pipeline writes these files under `artifacts/`:

- `scored_sample.csv`
- `presentation_examples.csv`
- `risk_map_language_platform.csv`
- `top_risk_segments.csv`
- `top_coordination_clusters.csv`
- `top_risk_authors.csv`
- `platform_normalized_risk.csv`
- `time_spikes.csv`
- `temporal_burst_windows.csv`
- `language_manipulative_share.csv`
- `case_studies.md`
- `nlp_text_model.npz`
- `nlp_text_model_metadata.json`
- `nlp_pseudo_label_training_summary.csv`
- `risk_map_language_platform.png`
- `language_manipulative_share.png`
- `top_suspicious_segments.png`
- `hourly_suspicious_share.png`
- `temporal_burst_windows.png`
- `risk_score_histogram.png`
- `platform_normalized_risk.png`
- `top_risk_authors.png`
- `run_summary.json`
- `pitch_deck.md`

## Scoring Logic

Final score is a weighted combination of:

- content risk: text length, repetition, links/hashtags/mentions, uppercase, punctuation, sentiment
- author risk: volume, burst posting, repetition, low topic diversity
- coordination risk: same narrative signature across multiple authors/platforms in compact time windows

Additional explainability artifacts:

- risk score histogram: shows how selective the scoring threshold is
- author-level summary: highlights authors with high volume, burst, and repetition signals
- cluster confidence score: combines known author spread, compact time window, cluster size, and exact narrative match
- platform-normalized risk: compares each platform's Review + High rate against the dataset average
- temporal burst windows: flags hourly Review + High or High share z-score spikes as campaign-burst evidence without changing row labels
- case studies: summarizes concrete jury-facing examples from real scored rows

Risk bands:

- `High`: `risk_score >= 0.65`; label is `Manipulative`
- `Review`: `0.45 <= risk_score < 0.65`; label remains `Organic`, but rows are included in dashboard review views
- `Low`: `risk_score < 0.45`

Very short non-empty replies such as `yes`, `lol`, or `No` are treated as low-evidence text instead of automatic manipulation. Empty/whitespace-only text remains high risk.
