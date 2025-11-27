#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
odp_summary_and_plots.py
One-file script that loads a CSV, produces plots, and writes textual/CSV/JSON summaries.

"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # use headless backend to avoid Qt/Wayland
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# ------------------------- Utilities -------------------------

def log(msg, level="INFO", want="INFO"):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if levels.index(level) >= levels.index(want):
        print(f"[{level}] {msg}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def slugify(name: str) -> str:
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    return re.sub(r"\s+", "_", stem)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # common misspelling normalization
    df = df.rename(columns={
        'ontolgoy_available?': 'ontology_available?',
        'odp_reused': 'odp_reused?',
        'odp_reused_?': 'odp_reused?',
    })
    return df

def coerce_booleans(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.upper()
    return s.replace({"TRUE": True, "FALSE": False, "YES": True, "NO": False, "1": True, "0": False})

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

# ------------------------- Core Work -------------------------

def load_and_clean(csv_path: Path, log_level: str) -> pd.DataFrame:
    log(f"Loading CSV: {csv_path}", want=log_level)
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    if 'year' in df.columns:
        df['year'] = coerce_numeric(df['year'])
    else:
        raise ValueError("Required column 'year' not found.")

    # Optional columns
    if 'odp_reused?' in df.columns:
        df['odp_reused?'] = coerce_booleans(df['odp_reused?'])
    if 'ontology_available?' in df.columns:
        df['ontology_available?'] = coerce_booleans(df['ontology_available?'])
    if 'type' not in df.columns:
        df['type'] = ""

    # Make some common fields present to avoid KeyErrors later
    for col in ['workshop', 'authors', 'citations']:
        if col not in df.columns:
            df[col] = np.nan

    # Clean citations numeric
    df['citations'] = pd.to_numeric(df['citations'], errors='coerce')

    return df

def filter_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)]

def series_to_full_year_index(s: pd.Series, start_year: int, end_year: int) -> pd.Series:
    idx = list(range(start_year, end_year + 1))
    return s.reindex(idx, fill_value=0)

def compute_stats(df: pd.DataFrame, start_year: int, end_year: int, log_level: str):
    # Basic per-year series
    papers_per_year = df.groupby('year').size()
    odp_per_year = df[df['type'].astype(str).str.contains("ODP", case=False, na=False)].groupby('year').size()
    odp_reused_per_year = df[
        (df.get('odp_reused?', False) == True) &
        (df['type'].astype(str).str.contains("ODP", case=False, na=False))
    ].groupby('year').size()

    # Align to a full year range
    papers_per_year = series_to_full_year_index(papers_per_year, start_year, end_year)
    odp_per_year = series_to_full_year_index(odp_per_year, start_year, end_year)
    odp_reused_per_year = series_to_full_year_index(odp_reused_per_year, start_year, end_year)

    stats = {
        "papers_per_year": papers_per_year.to_dict(),
        "odp_per_year": odp_per_year.to_dict(),
        "odp_reused_per_year": odp_reused_per_year.to_dict(),
        "total_odps": int(odp_per_year.sum()),
        "total_odp_reused": int(odp_reused_per_year.sum()),
    }

    # Workshops
    workshops = df['workshop'].dropna().astype(str).str.strip()
    by_workshop = workshops.value_counts().to_dict()
    stats["papers_per_workshop"] = by_workshop

    # Authors
    author_counter = Counter()
    for authors in df['authors'].dropna().astype(str):
        for a in re.split(r",| and ", authors):
            a = a.strip()
            if a:
                author_counter[a] += 1
    stats["top_authors"] = dict(author_counter.most_common(50))

    # Citations
    cits = df['citations'].dropna()
    stats["citation_total"] = int(cits.sum()) if len(cits) else 0
    stats["citation_avg"] = float(cits.mean()) if len(cits) else 0.0
    stats["citation_median"] = float(cits.median()) if len(cits) else 0.0

    log("Computed statistics.", want=log_level)
    return stats, papers_per_year, odp_per_year, odp_reused_per_year

# ------------------------- Plotting -------------------------

def plot_wop_stats(papers_per_year, odp_per_year, odp_reused_per_year, fig_path: Path, log_level: str):
    plt.figure(figsize=(14, 7))
    ax = papers_per_year.plot(kind='bar', label='Total Papers', alpha=0.9)
    odp_per_year.plot(kind='bar', label='ODPs Published', alpha=0.7, ax=ax)
    odp_reused_per_year.plot(kind='bar', label='ODP Reused', alpha=0.5, ax=ax)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(str(int(h)), (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontsize=8)

    ax.set_title('WOP Paper Statistics per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Saved: {fig_path}", want=log_level)

def plot_ontology_vs_odp(df: pd.DataFrame, start_year: int, end_year: int, fig_path: Path, log_level: str):
    ontology_per_year = df[df['type'].astype(str).str.contains("ontology", case=False, na=False)].groupby('year').size()
    odp_pub_per_year = df[df['type'].astype(str).str.contains("odp", case=False, na=False)].groupby('year').size()

    years = list(range(start_year, end_year + 1))
    x = np.arange(len(years))
    bar_width = 0.4
    ontology_counts = [int(ontology_per_year.get(y, 0)) for y in years]
    odp_counts = [int(odp_pub_per_year.get(y, 0)) for y in years]

    plt.figure(figsize=(14, 7))
    plt.bar(x - bar_width/2, ontology_counts, width=bar_width, label='Ontology Papers')
    plt.bar(x + bar_width/2, odp_counts, width=bar_width, label='ODP Papers')

    for i, val in enumerate(ontology_counts):
        if val > 0:
            plt.annotate(str(val), (x[i] - bar_width/2, val), ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
    for i, val in enumerate(odp_counts):
        if val > 0:
            plt.annotate(str(val), (x[i] + bar_width/2, val), ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    plt.xticks(x, years, rotation=45)
    plt.title('Ontology vs ODP Introduction per Year (Side-by-Side)')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Saved: {fig_path}", want=log_level)

def plot_ontology_availability(df: pd.DataFrame, fig_path: Path, log_level: str):
    if 'ontology_available?' not in df.columns:
        log("No 'ontology_available?' column; skipping availability plot.", want=log_level)
        return

    s = df[df['ontology_available?'] == True].groupby('year').size().sort_index()

    if s.empty:
        log("No rows with ontology_available? == True; skipping availability plot.", want=log_level)
        return

    plt.figure(figsize=(14, 7))
    ax = s.plot(kind='bar', label='Ontologies Available')
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(str(int(h)), (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
    ax.set_title('Ontologies Available per Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Saved: {fig_path}", want=log_level)

def plot_citations_hist(df: pd.DataFrame, fig_path: Path, log_level: str):
    if 'citations' not in df.columns:
        return
    c = df['citations'].dropna()
    if not len(c):
        return
    plt.figure(figsize=(12, 6))
    plt.hist(c, bins=30)
    plt.title('Citations Distribution')
    plt.xlabel('Citations')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Saved: {fig_path}", want=log_level)

# ------------------------- Outputs -------------------------

def write_text_summary(stats: dict, out_path: Path, top_n: int, log_level: str):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("📊 Paper Statistics Summary\n")
        f.write("=" * 40 + "\n\n")

        # Totals
        f.write(f"Total ODPs: {stats['total_odps']}\n")
        f.write(f"Total ODP reused: {stats['total_odp_reused']}\n")
        f.write(f"Total citations (if any): {stats.get('citation_total', 0)}\n")
        f.write(f"Average citations per paper: {stats.get('citation_avg', 0.0):.2f}\n")
        f.write(f"Median citations per paper: {stats.get('citation_median', 0.0):.2f}\n")

        # Per year
        f.write("\n📅 Papers per year:\n")
        for y, c in sorted(stats['papers_per_year'].items()):
            f.write(f"  {y}: {c}\n")

        f.write("\n📦 ODPs per year:\n")
        for y, c in sorted(stats['odp_per_year'].items()):
            f.write(f"  {y}: {c}\n")

        f.write("\n🔁 ODP reused per year:\n")
        for y, c in sorted(stats['odp_reused_per_year'].items()):
            f.write(f"  {y}: {c}\n")

        # Workshops
        f.write("\n🏷️ Papers per workshop:\n")
        for w, c in sorted(stats.get("papers_per_workshop", {}).items()):
            f.write(f"  {w}: {c}\n")

        # Authors
        f.write(f"\n🧑‍🔬 Top {top_n} most frequent authors:\n")
        for author, c in list(stats.get("top_authors", {}).items())[:top_n]:
            f.write(f"  {author}: {c} paper(s)\n")

    log(f"Wrote text summary: {out_path}", want=log_level)

def write_machine_summaries(stats: dict, data_dir: Path, log_level: str):
    # JSON
    json_path = data_dir / "statistics_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    # CSV convenience exports
    pd.Series(stats["papers_per_year"]).rename("count").to_csv(data_dir / "papers_per_year.csv", header=True)
    pd.Series(stats["odp_per_year"]).rename("count").to_csv(data_dir / "odp_per_year.csv", header=True)
    pd.Series(stats["odp_reused_per_year"]).rename("count").to_csv(data_dir / "odp_reused_per_year.csv", header=True)
    log(f"Wrote JSON/CSV summaries under: {data_dir}", want=log_level)

# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate plots and summary from a CSV (WOP/ODP).")
    parser.add_argument("--input", required=True, help="Input CSV file (e.g., 'input/wop.csv').")
    parser.add_argument("--data-dir", required=True, help="Output data directory (e.g., 'output/wop').")
    parser.add_argument("--fig-dir", default=None, help="Figures directory (defaults to DATA_DIR/figures).")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else data_dir / "figures"
    logs_dir = data_dir / "logs"

    for p in [data_dir, fig_dir, logs_dir]:
        ensure_dir(p)

    # Load & clean
    df = load_and_clean(Path(args.input), log_level=args.log_level)

    # Filter by year window
    df_f = filter_years(df, args.start_year, args.end_year)

    # Compute
    stats, papers_per_year, odp_per_year, odp_reused_per_year = compute_stats(
        df_f, args.start_year, args.end_year, args.log_level
    )

    # Plots
    plot_wop_stats(
        papers_per_year, odp_per_year, odp_reused_per_year,
        fig_dir / "wop_stats_per_year.png", args.log_level
    )
    plot_ontology_vs_odp(df_f, args.start_year, args.end_year, fig_dir / "ontology_vs_odp.png", args.log_level)
    plot_ontology_availability(df_f, fig_dir / "ontology_available_per_year.png", args.log_level)
    plot_citations_hist(df_f, fig_dir / "citations_hist.png", args.log_level)

    # Summaries
    write_text_summary(stats, data_dir / "statistics_summary.txt", args.top_n, args.log_level)
    write_machine_summaries(stats, data_dir, args.log_level)

    print("✔ Done.")

if __name__ == "__main__":
    main()
