# Report Build Instructions

This directory contains an ACM-style LaTeX report for the ESG News Risk Forecaster project.

## Files

- `main.tex`: Main ACM SIGCONF-style report.
- `references.bib`: BibTeX references for datasets, models, tools, and related work.
- `figures/`: Report-local figures copied or generated from repository outputs.
- `tables/`: LaTeX tables and summary macros generated from repository outputs.

## Compile Locally

Run these commands from the `report/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The expected output is `main.pdf`.

The source prefers `acmart` when that class is available. If local TeX does not include `acmart.cls`, `main.tex` falls back to a two-column article layout so the report still compiles. On Overleaf or a full TeX Live install with the ACM package, it will use `\documentclass[sigconf,authordraft]{acmart}`.

## Notes

The figures and tables were generated from the existing repository artifacts in `data/processed`, `data/interim`, and `outputs/`. The report does not invent results; reported metrics are taken from the generated model evaluation CSVs.
