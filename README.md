# SPC I‑MR Dashboard + PDF Report (Portfolio Project)

This repository demonstrates a **regulated-industry friendly** workflow for manufacturing / batch data:
- Import data from CSV (sanitized/demo)
- Build **SPC Individuals + Moving Range (I‑MR)** charts
- Apply simple signal rules (e.g., **beyond ±3σ**, **run rule: 8 points on one side**)
- Export an **audit-style PDF report** including an investigation list (flagged points)

## Why this project is relevant (Pharma / MedTech)
- Shows a **quality mindset** (spec limits, traceability of rules, investigation list)
- Demonstrates **data handling + reporting** with Python (pandas, matplotlib)
- Fits typical tasks in **Process Engineering / MSAT / Validation / Quality Engineering**

## Repository structure
- `src/spc_dashboard.py` – report generator
- `data/sample_process_data.csv` – demo dataset
- `config/config.json` – parameter selection, spec limits, rule toggles
- `docs/SPC_Report_MT-01_Demo.pdf` – example output

## Run locally
Requirements: Python 3.10+

```bash
pip install pandas numpy matplotlib
python src/spc_dashboard.py --input data/sample_process_data.csv --config config/config.json --output docs/SPC_Report.pdf
```

## Data format (CSV)
Columns:
- `timestamp` (YYYY-MM-DD)
- `batch_id`
- `parameter`
- `value`
- `unit`

## Disclaimer
This is an **educational/demo** project. The dataset is synthetic/sanitized and contains **no confidential company data**.
