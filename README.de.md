# SPC I‑MR Dashboard + PDF‑Report (Portfolio‑Projekt)

Dieses Repository zeigt einen **regulierten, praxisnahen** Workflow für Produktions-/Batchdaten:
- Import von CSV‑Daten (synthetisch/anonmyisiert)
- Erstellung von **SPC‑Regelkarten Individuals + Moving Range (I‑MR)**
- Anwendung einfacher Signale (z. B. **außerhalb ±3σ**, **Run‑Rule: 8 Punkte auf einer Seite**)
- Export eines **audit‑ähnlichen PDF‑Reports** inkl. Untersuchungsliste (auffällige Punkte)

## Relevanz (Pharma / MedTech)
- Demonstriert **Quality‑Mindset** (Spezifikationsgrenzen, Regeln, Untersuchungsliste)
- Zeigt **Datenaufbereitung + Reporting** mit Python (pandas, matplotlib)
- Passt zu Aufgaben in **Prozesstechnik / MSAT / Validierung / Qualität**

## Struktur
- `src/spc_dashboard.py` – Report‑Generator
- `data/sample_process_data.csv` – Demo‑Datensatz
- `config/config.json` – Parameter, Spezifikationen, Regel‑Schalter
- `docs/SPC_Report_MT-01_Demo.pdf` – Beispiel‑Output

## Lokal ausführen
Voraussetzung: Python 3.10+

```bash
pip install pandas numpy matplotlib
python src/spc_dashboard.py --input data/sample_process_data.csv --config config/config.json --output docs/SPC_Report.pdf
```

## Datenformat (CSV)
Spalten:
- `timestamp` (YYYY-MM-DD)
- `batch_id`
- `parameter`
- `value`
- `unit`

## Hinweis / Disclaimer
Dies ist ein **Lern-/Demo‑Projekt**. Der Datensatz ist synthetisch/anonmyisiert und enthält **keine vertraulichen Unternehmensdaten**.
