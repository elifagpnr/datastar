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

If your system Python says `externally-managed-environment`, use the virtualenv commands above. In VS Code/Jupyter, select `.venv/bin/python` as the notebook kernel before running `ads.ipynb`.

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
- `risk_map_language_platform.png`
- `top_suspicious_segments.png`
- `hourly_suspicious_share.png`
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

Risk bands:

- `High`: `risk_score >= 0.65`; label is `Manipulative`
- `Review`: `0.45 <= risk_score < 0.65`; label remains `Organic`, but rows are included in dashboard review views
- `Low`: `risk_score < 0.45`

Very short non-empty replies such as `yes`, `lol`, or `No` are treated as low-evidence text instead of automatic manipulation. Empty/whitespace-only text remains high risk.
