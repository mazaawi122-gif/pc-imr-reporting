#!/usr/bin/env python3
"""
SPC Dashboard + PDF Report Export (I-MR chart)
------------------------------------------------
This script reads a CSV with columns:
  timestamp,batch_id,parameter,value,unit

It generates:
  - Individuals chart (I)
  - Moving Range chart (MR)
  - Histogram with spec limits
  - A PDF report with an investigation list (flagged points)

Usage:
  python spc_dashboard.py --input sample_process_data.csv --config config.json --output SPC_Report.pdf
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def moving_range(x):
    x = np.asarray(x)
    return np.abs(np.diff(x))

def imr_limits(x):
    x = np.asarray(x)
    mr = moving_range(x)
    xbar = np.mean(x)
    mrbar = np.mean(mr) if len(mr) else 0.0
    d2 = 1.128  # for MR(2)
    sigma_hat = mrbar / d2 if d2 != 0 else np.nan
    ucl = xbar + 3 * sigma_hat
    lcl = xbar - 3 * sigma_hat
    # MR chart limits
    D4 = 3.267
    D3 = 0.0
    mr_ucl = D4 * mrbar
    mr_lcl = D3 * mrbar
    return {"xbar": xbar, "mrbar": mrbar, "sigma_hat": sigma_hat,
            "I_UCL": ucl, "I_LCL": lcl, "MR_UCL": mr_ucl, "MR_LCL": mr_lcl}

def rule_beyond_3sigma(x, lcl, ucl):
    x = np.asarray(x)
    return (x > ucl) | (x < lcl)

def rule_run_8_one_side(x, center):
    x = np.asarray(x)
    above = x > center
    below = x < center
    flags = np.zeros(len(x), dtype=bool)
    run = 0
    last = None
    for i in range(len(x)):
        cur = True if above[i] else (False if below[i] else None)
        if cur is None:
            run = 0
            last = None
            continue
        if last is None or cur == last:
            run += 1
        else:
            run = 1
        last = cur
        if run >= 8:
            flags[i] = True
    for i in range(len(x)):
        if flags[i]:
            flags[max(0, i-7):i+1] = True
    return flags

def capability_indices(x, lsl, usl, sigma_hat, mean):
    if sigma_hat is None or np.isnan(sigma_hat) or sigma_hat == 0:
        return {"Cp": np.nan, "Cpk": np.nan}
    cp = (usl - lsl) / (6 * sigma_hat)
    cpu = (usl - mean) / (3 * sigma_hat)
    cpl = (mean - lsl) / (3 * sigma_hat)
    cpk = min(cpu, cpl)
    return {"Cp": cp, "Cpk": cpk, "Cpu": cpu, "Cpl": cpl}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV file with process/test data")
    ap.add_argument("--config", required=True, help="JSON config (title/spec limits/rules)")
    ap.add_argument("--output", required=True, help="Output PDF report path")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Filter to parameter if multiple exist
    if "parameter" in cfg and "parameter" in df.columns:
        df = df[df["parameter"] == cfg["parameter"]].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    vals = df["value"].to_numpy()

    limits = imr_limits(vals)
    lsl = cfg["spec_limits"]["LSL"]
    usl = cfg["spec_limits"]["USL"]
    cap = capability_indices(vals, lsl, usl, limits["sigma_hat"], limits["xbar"])

    flag_3s = rule_beyond_3sigma(vals, limits["I_LCL"], limits["I_UCL"]) if cfg["run_rules"].get("beyond_3sigma", True) else np.zeros(len(vals), bool)
    flag_run8 = rule_run_8_one_side(vals, limits["xbar"]) if cfg["run_rules"].get("run_8_one_side", True) else np.zeros(len(vals), bool)
    flags = flag_3s | flag_run8

    with PdfPages(args.output) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(cfg.get("title", "SPC Report"), fontsize=16, y=0.97)
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.axis("off")
        lines = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"Parameter: {cfg.get('parameter','(not set)')} ({cfg.get('unit','')})",
            f"Spec limits: LSL={lsl:.3f}, USL={usl:.3f}",
            "",
            "Key Statistics",
            f"• N = {len(vals)}",
            f"• Mean = {limits['xbar']:.3f}",
            f"• Estimated σ (from MR) = {limits['sigma_hat']:.3f}",
            "",
            "Capability (estimated)",
            f"• Cp = {cap['Cp']:.2f}",
            f"• Cpk = {cap['Cpk']:.2f}",
            "",
            "SPC Signals",
            f"• Beyond ±3σ points: {int(flag_3s.sum())}",
            f"• Run of 8 on one side: {int(flag_run8.sum())}",
            f"• Total flagged points: {int(flags.sum())}",
        ]
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        # Individuals chart
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        x = np.arange(1, len(vals)+1)
        ax.plot(x, vals, marker="o", linewidth=1)
        ax.axhline(limits["xbar"], linestyle="--")
        ax.axhline(limits["I_UCL"], linestyle="--")
        ax.axhline(limits["I_LCL"], linestyle="--")
        ax.axhline(usl, linestyle=":")
        ax.axhline(lsl, linestyle=":")
        ax.scatter(x[flags], vals[flags], s=60, marker="o")
        ax.set_title("Individuals Chart (I)")
        ax.set_xlabel("Observation"); ax.set_ylabel(f"Value ({cfg.get('unit','')})")
        ax.grid(True, alpha=0.25)
        pdf.savefig(fig); plt.close(fig)

        # MR chart
        mr = moving_range(vals)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        x_mr = np.arange(2, len(vals)+1)
        ax.plot(x_mr, mr, marker="o", linewidth=1)
        ax.axhline(limits["mrbar"], linestyle="--")
        ax.axhline(limits["MR_UCL"], linestyle="--")
        ax.axhline(limits["MR_LCL"], linestyle="--")
        ax.set_title("Moving Range Chart (MR)")
        ax.set_xlabel("Observation"); ax.set_ylabel(f"Moving Range ({cfg.get('unit','')})")
        ax.grid(True, alpha=0.25)
        pdf.savefig(fig); plt.close(fig)

        # Histogram
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.hist(vals, bins=18)
        ax.axvline(lsl, linestyle="--"); ax.axvline(usl, linestyle="--")
        ax.set_title("Histogram with Specification Limits")
        ax.set_xlabel(f"Value ({cfg.get('unit','')})"); ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
        pdf.savefig(fig); plt.close(fig)

        # Flag list
        flagged_df = df.loc[flags, ["timestamp", "batch_id", "value"]].copy()
        flagged_df["rule"] = np.where(flag_3s[flags], "Beyond 3σ", "Run of 8")
        both = flag_3s & flag_run8
        if both.any():
            flagged_df.loc[both[flags], "rule"] = "Beyond 3σ + Run of 8"

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Flagged Points (Investigation List)", fontsize=14, y=0.97)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88]); ax.axis("off")
        if flagged_df.empty:
            ax.text(0, 1, "No points flagged by the selected rules.", va="top", fontsize=11)
        else:
            rows = ["Date       | Batch   | Value | Rule",
                    "-----------|---------|-------|-----------------------"]
            for _, r in flagged_df.head(50).iterrows():
                rows.append(f"{r['timestamp']} | {r['batch_id']} | {float(r['value']):.3f} | {r['rule']}")
            ax.text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

if __name__ == "__main__":
    main()
