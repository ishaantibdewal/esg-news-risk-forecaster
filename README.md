ESG2Risk-Inspired Volatility Forecasting
========================================

This DSC 148 project reproduces and extends the ESG2Risk paper idea with
FNSPID financial news and price history. The core question is whether
ESG-related financial news improves future realized volatility prediction
beyond price-history-only and sentiment-only baselines.

Key implementation rules:

- Do not load `data/raw/fnspid/nasdaq_external_data.csv` fully with
  `pd.read_csv(news_path)`.
- Inspect the news file only with bounded reads such as `nrows=1000`.
- Process news with chunks, ticker/date filters, and Parquet outputs.
- Keep raw, interim, processed, model, and large output files out of git.

Main pipeline:

1. Safely inspect raw news and price schemas.
2. Validate the handpicked stock universe and benchmark.
3. Filter FNSPID news by ticker/date in chunks.
4. Clean price histories and build future volatility labels.
5. Build weekly price, ESG keyword, FinBERT, and ClimateBERT features.
6. Merge a weekly modeling panel.
7. Compare price-only, ESG-keyword, FinBERT, ClimateBERT, combined, and full
   models with time-aware validation.
8. Evaluate with regression/classification metrics and ESG2Risk-style
   predicted-risk quintile analysis.

Reusable implementation code lives in `src/`. Notebooks in `notebooks/` should
call those modules rather than duplicating pipeline logic.

Recommended command order:

```bash
./.venv/bin/python -m src.pipeline setup
./.venv/bin/python -m src.pipeline inspect --nrows 1000 --scan-news
./.venv/bin/python -m src.pipeline filter-news --chunksize 100000
./.venv/bin/python -m src.pipeline prices-labels
./.venv/bin/python -m src.pipeline features
./.venv/bin/python -m src.pipeline finbert --batch-size 16
./.venv/bin/python -m src.pipeline climatebert --batch-size 16
./.venv/bin/python -m src.pipeline panel
./.venv/bin/python -m src.pipeline model-eval
./.venv/bin/python -m src.pipeline report
```

The transformer commands intentionally score only ESG keyword-filtered articles
by default. Use `--all-news` only for a larger compute run.
