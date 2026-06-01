# ESG News Risk Forecaster

This DSC 148 project reproduces and extends the ESG2Risk idea: can ESG-sensitive
financial news improve short-horizon stock risk forecasts beyond market-only
baselines?

The pipeline builds a weekly ticker panel from FNSPID news and price histories,
engineers price, ESG keyword, FinBERT sentiment, and ClimateBERT relevance
features, then evaluates volatility regression and high-volatility
classification with chronological train/validation/test splits.

## Repository Contents

- `src/`: reusable pipeline code for loading, filtering, feature generation,
  labeling, modeling, evaluation, and visualization.
- `notebooks/results_report.ipynb`: presentation-oriented notebook walkthrough of
  the generated results.
- `report/main.tex` and `report/main.pdf`: ACM-style final report.
- `report/README.md`: local report build instructions.
- `outputs/reports/results_summary.md`: generated metric summary.
- `outputs/tables/` and `outputs/figures/`: generated result tables and plots.

Large raw/intermediate/model artifacts are intentionally not tracked. The raw
FNSPID news CSV is about 23 GB and the full local `data/` directory is much
larger than a normal GitHub submission.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch` installation needs platform-specific handling, install the
appropriate build from the PyTorch instructions first, then rerun
`pip install -r requirements.txt`.

## Data Layout

To fully reproduce the pipeline, place FNSPID under:

```text
data/raw/fnspid/
  nasdaq_external_data.csv
  full_history/
    AAPL.csv
    MSFT.csv
    ...
```

The code never loads the 23 GB news file with an unbounded `pd.read_csv`.
Inspection uses bounded reads, and production filtering uses chunks.

## Full Reproduction

Run commands from the repository root after the raw data is in place:

```bash
python -m src.pipeline setup
python -m src.pipeline inspect --nrows 1000 --scan-news
python -m src.pipeline filter-news --chunksize 100000
python -m src.pipeline prices-labels
python -m src.pipeline features
python -m src.pipeline finbert --batch-size 16
python -m src.pipeline climatebert --batch-size 16
python -m src.pipeline panel
python -m src.pipeline model-eval
python -m src.pipeline tuned-model-eval
python -m src.pipeline report
```

The transformer stages score ESG keyword-filtered articles by default to keep
runtime manageable. Use `--all-news` only for a larger compute run. For a smoke
test of the transformer commands, add `--max-articles 100`.

## Colab Demo

The primary demo is the Colab-ready results notebook:

<https://colab.research.google.com/github/ishaantibdewal/esg-news-risk-forecaster/blob/main/notebooks/results_report.ipynb>

Run the notebook cells from top to bottom. It is designed to work directly from
the GitHub repository outputs, without the 23 GB raw FNSPID CSV and without
local Parquet intermediates.

```bash
jupyter notebook notebooks/results_report.ipynb
```

The notebook walks through dataset construction, EDA, chronological splits,
tuned results, ablations, diagnostics, saved prediction examples, and final
interpretation. This is the clearest "paper results" style demo for grading.

## Main Generated Results

The reported test panel contains 5,852 ticker-weeks across 14 tickers from
2016-2023. The tuned 2023 test results show:

- Price-only remains a strong baseline.
- Price plus ClimateBERT is the best selected combined regression setup with
  RMSE near 0.0063 and `R^2` near 0.52.
- Tuned classification reaches ROC-AUC near 0.91, with climate-aware text adding
  modest incremental value in the selected combined model.
- ESG/news features are useful as incremental risk signals, not as substitutes
  for market history.

See `outputs/reports/results_summary.md`, `outputs/tables/`, and `report/main.pdf`
for the full numbers.

## Project Criteria Mapping

- Dataset and EDA: FNSPID news plus price histories; raw inventory, ticker/year
  coverage, target distributions, ESG category diagnostics, and coverage tables.
- Predictive task: 21-trading-day realized volatility regression and
  high-volatility classification with chronological splits.
- Baselines: market-only models, news-only models, ESG keyword models, FinBERT,
  ClimateBERT, combined feature groups, and full feature groups.
- Model: ridge/logistic baselines plus tree ensembles and tuned histogram
  gradient boosting/random forest configurations using engineered price and text
  features.
- Literature: ESG2Risk, FNSPID, financial NLP, FinBERT, ClimateBERT, volatility
  forecasting, and related text-risk work are discussed in the report.
- Results: model comparisons, ablations, tuned-vs-untuned checks, bootstrap delta
  checks, leakage checks, risk quintile analysis, and error/feature diagnostics.
- GitHub reproducibility: this README gives environment setup, data layout,
  staged commands, Colab notebook usage, and report build location.
