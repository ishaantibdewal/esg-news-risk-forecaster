# Submission Checklist

## What to Submit

- GitHub repository: `https://github.com/ishaantibdewal/esg-news-risk-forecaster`
- Final report PDF: `report/main.pdf`
- Colab demo notebook: `notebooks/results_report.ipynb`
- Generated result artifacts: `outputs/tables/`, `outputs/figures/`, and `outputs/reports/results_summary.md`
- Reusable source code: `src/`
- Report source and references: `report/main.tex`, `report/references.bib`, `report/tables/`, `report/figures/`

## Do Not Submit

- `data/raw/`, `data/interim/`, `data/processed/`
- `outputs/models/`
- `.venv/`
- `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`
- LaTeX temporary files such as `*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`

These are excluded by `.gitignore` or should remain local only.

## Data Required for Full Reproduction

Place FNSPID under:

```text
data/raw/fnspid/
  nasdaq_external_data.csv
  full_history/
    AAPL.csv
    MSFT.csv
    ...
```

The raw news CSV is about 23 GB and is intentionally excluded from GitHub.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch installation fails on a fresh machine, install the platform-specific
PyTorch build first, then rerun `pip install -r requirements.txt`.

## Full Pipeline Reproduction

Run from the repository root after the raw data is in place:

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

Transformer scoring defaults to ESG keyword-filtered articles. For a quick
smoke test, add `--max-articles 100` to `finbert` and `climatebert`.

## Colab Demo

Open:

```text
https://colab.research.google.com/github/ishaantibdewal/esg-news-risk-forecaster/blob/main/notebooks/results_report.ipynb
```

Run all cells from top to bottom. The notebook uses tracked generated artifacts,
so graders do not need the raw FNSPID data to view the results walkthrough.

## Report Build

From `report/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Expected output: `report/main.pdf`.

## Project Requirements Mapping

- Dataset and EDA: `report/main.tex`, `notebooks/results_report.ipynb`, `outputs/tables/raw_data_inventory.csv`, `outputs/tables/news_coverage_by_ticker_year.csv`
- Predictive task: 21-trading-day realized volatility regression and high-volatility classification in `src/label_building.py` and report Sections 4-5
- Models and baselines: price-only, news-only, ESG keyword, FinBERT, ClimateBERT, combined, and full feature groups in `src/modeling.py` and `src/pipeline.py`
- Literature: `report/references.bib` and Related Work section
- Results: `outputs/tables/model_metrics.csv`, `outputs/tables/tuned_model_metrics.csv`, `outputs/tables/tuned_vs_untuned.csv`, `outputs/tables/predicted_risk_quintiles.csv`, report Results section
- Reproducibility: `README.md`, this checklist, and staged pipeline commands

## Known Limitations

- Raw news data is too large for GitHub and must be downloaded separately.
- Transformer scoring was run on ESG keyword-filtered articles, not the full 23 GB news file.
- The ticker universe is small and biased toward large, news-covered equities.
- Ticker-symbol entity linking and ESG keyword matching are noisy.
- Results are predictive, not causal.
- Weekly aggregation may blur exact event timing.

## Final Commands Before Submission

```bash
python3 -m compileall src
MPLCONFIGDIR=/private/tmp/mplconfig MPLBACKEND=Agg python3 - <<'PY'
import json
from pathlib import Path
json.loads(Path('notebooks/results_report.ipynb').read_text())
print('notebook_json_ok')
PY
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..
git status --short
```

