#!/usr/bin/env python3
"""
make_odp_figures.py

Generate publication-ready descriptive statistics and plots from the
ODP-reuse pipeline outputs. Reads per-paper "*-classified.csv" files
under data/<year>/ and (optionally) "odp_reuse_summary.csv".

Figures are saved into a "figures/" directory. If a plot is not possible
(e.g., insufficient data), a descriptive TXT file is saved instead.

Requirements:
  - Python 3.8+
  - matplotlib (no seaborn)
  - Standard library only otherwise

Usage:
  python make_odp_figures.py \
    --data-dir data \
    --summary-file odp_reuse_summary.csv \
    --fig-dir figures \
    --top-n 20 \
    --start-year 2000 \
    --end-year 2100
"""

import argparse
import csv
import os
import math
from collections import Counter, defaultdict
from statistics import mean, median
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

# ----------------------- IO Helpers -----------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_txt(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def list_classified_csvs(root: str):
    files = []
    if not os.path.isdir(root):
        return files
    for entry in os.scandir(root):
        if entry.is_dir():
            for f in os.scandir(entry.path):
                if f.is_file() and f.name.endswith("-classified.csv"):
                    files.append(f.path)
        elif entry.is_file() and entry.name.endswith("-classified.csv"):
            files.append(entry.path)
    files.sort()
    return files

def read_classified_rows(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            rows.append(r)
    return rows, reader.fieldnames if rows else []

def read_summary_rows(path: str):
    if not path or not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for r in reader:
            rows.append(r)
    return rows

# ----------------------- Plot Helpers -----------------------
def save_bar(x_labels, counts, title, xlabel, ylabel, out_path, rotate=45):
    if not x_labels or not counts or sum(counts) == 0:
        write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': no data.")
        return
    plt.figure(figsize=(10, 6), dpi=150)
    xs = range(len(x_labels))
    plt.bar(xs, counts)
    plt.xticks(xs, x_labels, rotation=rotate, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_pie(labels, sizes, title, out_path):
    if not labels or not sizes or sum(sizes) == 0:
        write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': no data.")
        return
    plt.figure(figsize=(6, 6), dpi=150)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_line(x_vals, y_vals, title, xlabel, ylabel, out_path):
    if not x_vals or not y_vals or len(x_vals) != len(y_vals) or sum(y_vals) == 0:
        write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': no data.")
        return
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_hist(values, bins, title, xlabel, ylabel, out_path):
    if not values:
        write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': no data.")
        return
    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_stacked_bars(categories, stacks_dict, title, xlabel, ylabel, out_path, rotate=45):
    """
    categories: list of base categories on x-axis (e.g., years)
    stacks_dict: dict stack_name -> list of counts aligned with categories
    """
    if not categories or not stacks_dict:
        write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': no data.")
        return
    # validate lengths
    n = len(categories)
    for v in stacks_dict.values():
        if len(v) != n:
            write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': misaligned stack lengths.")
            return
    if sum(sum(v) for v in stacks_dict.values()) == 0:
        write_txt(out_path.replace(".png", ".txt"), f"Cannot plot '{title}': all zeros.")
        return
    plt.figure(figsize=(10, 6), dpi=150)
    xs = range(n)
    bottom = [0]*n
    for name, vals in stacks_dict.items():
        plt.bar(xs, vals, bottom=bottom, label=name)
        bottom = [bottom[i] + vals[i] for i in range(n)]
    plt.xticks(xs, categories, rotation=rotate, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# ----------------------- Main Logic -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Root directory containing *-classified.csv under year folders")
    ap.add_argument("--summary-file", default="odp_reuse_summary.csv", help="Optional summary CSV path")
    ap.add_argument("--fig-dir", default="figures", help="Where to save figures and text outputs")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N items for rank plots (ODP names, venues)")
    ap.add_argument("--start-year", type=int, default=1900, help="Lower bound filter for citing_year series")
    ap.add_argument("--end-year", type=int, default=2100, help="Upper bound filter for citing_year series")
    args = ap.parse_args()

    ensure_dir(args.fig_dir)

    # 1) Collect all classified rows
    classified_files = list_classified_csvs(args.data_dir)
    if not classified_files:
        write_txt(os.path.join(args.fig_dir, "NO_DATA.txt"),
                  f"No *-classified.csv files found under '{args.data_dir}'. "
                  "Run the classifier first.")
        return

    all_rows = []
    for f in classified_files:
        rows, _ = read_classified_rows(f)
        all_rows.extend(rows)

    # Basic counts
    labels = [(r.get("odpreuse_label") or "").strip() for r in all_rows]
    label_counts = Counter([l for l in labels if l])
    save_pie(
        labels=list(label_counts.keys()),
        sizes=list(label_counts.values()),
        title="Overall ODP Reuse Labels",
        out_path=os.path.join(args.fig_dir, "fig1_labels_pie.png")
    )

    # 2) Time series: Likely by citing_year
    def as_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    ts_counts = Counter()
    for r in all_rows:
        cy = as_int(r.get("citing_year"))
        if cy is None or cy < args.start_year or cy > args.end_year:
            continue
        if (r.get("odpreuse_label") or "").strip().lower() == "likely":
            ts_counts[cy] += 1
    if ts_counts:
        years = sorted(ts_counts.keys())
        counts = [ts_counts[y] for y in years]
    else:
        years, counts = [], []
    save_line(
        x_vals=years, y_vals=counts,
        title="Likely ODP Reuse Over Time (by citing year)",
        xlabel="Citing Year", ylabel="Count",
        out_path=os.path.join(args.fig_dir, "fig2_likely_time_series.png")
    )

    # 3) Top ODP names mentioned
    name_counter = Counter()
    for r in all_rows:
        names = (r.get("matched_odp_names") or "").split(";")
        for n in names:
            n = n.strip()
            if n:
                name_counter[n] += 1
    if name_counter:
        top_names = name_counter.most_common(args.top_n)
        save_bar(
            x_labels=[k for k,_ in top_names],
            counts=[v for _,v in top_names],
            title=f"Top {args.top_n} ODP Names Detected",
            xlabel="ODP Name", ylabel="Mentions",
            out_path=os.path.join(args.fig_dir, "fig3_top_odp_names.png"),
            rotate=60
        )
    else:
        write_txt(os.path.join(args.fig_dir, "fig3_top_odp_names.txt"),
                  "No matched ODP names found.")

    # 4) Open Access availability (proxy: abstract fetched)
    oa_counts = Counter()
    for r in all_rows:
        val = str(r.get("abstract_fetched") or "").strip()
        if val in ("1", "True", "true"):
            oa_counts["Abstract Available"] += 1
        else:
            oa_counts["No Abstract"] += 1
    save_bar(
        x_labels=list(oa_counts.keys()),
        counts=list(oa_counts.values()),
        title="Abstract Availability (via OpenAlex)",
        xlabel="", ylabel="Count",
        out_path=os.path.join(args.fig_dir, "fig4_abstract_availability.png"),
        rotate=0
    )

    # 5) Stacked bars: Label distribution by citing year
    by_year_label = defaultdict(Counter)
    all_years = set()
    for r in all_rows:
        cy = as_int(r.get("citing_year"))
        if cy is None or cy < args.start_year or cy > args.end_year:
            continue
        lab = (r.get("odpreuse_label") or "").strip()
        if lab:
            by_year_label[cy][lab] += 1
            all_years.add(cy)
    if all_years:
        cats = sorted(all_years)
        labs = ["Likely", "Possible", "Unlikely"]
        stacks = {lab: [by_year_label[y][lab] for y in cats] for lab in labs}
        save_stacked_bars(
            categories=cats, stacks_dict=stacks,
            title="Label Distribution by Citing Year",
            xlabel="Citing Year", ylabel="Count",
            out_path=os.path.join(args.fig_dir, "fig5_labels_by_year.png"),
            rotate=0
        )
    else:
        write_txt(os.path.join(args.fig_dir, "fig5_labels_by_year.txt"),
                  "Insufficient citing_year data to build stacked bars.")

    # 6) Histogram of citing counts per seed (from summary, if present)
    summary_rows = read_summary_rows(args.summary_file)
    citing_counts = []
    for r in summary_rows:
        try:
            citing_counts.append(int(r.get("num_citing", "0")))
        except Exception:
            pass
    if citing_counts:
        bins = min(20, max(5, int(math.sqrt(len(citing_counts)))))
        save_hist(
            values=citing_counts, bins=bins,
            title="Histogram: Number of Citing Works per Seed Paper",
            xlabel="# Citing Works", ylabel="Seed Papers",
            out_path=os.path.join(args.fig_dir, "fig6_hist_citing_counts.png")
        )
        # Save basic stats
        stats_path = os.path.join(args.fig_dir, "stats_citing_counts.txt")
        lines = [
            f"N seeds: {len(citing_counts)}",
            f"Mean: {mean(citing_counts):.2f}",
            f"Median: {median(citing_counts):.2f}",
            f"Min: {min(citing_counts)}",
            f"Max: {max(citing_counts)}",
        ]
        write_txt(stats_path, "\n".join(lines))
    else:
        write_txt(os.path.join(args.fig_dir, "fig6_hist_citing_counts.txt"),
                  "No summary file or 'num_citing' values to plot histogram.")

    # 7) Top venues
    venue_counter = Counter()
    for r in all_rows:
        v = (r.get("venue_name_openalex") or "").strip()
        if v:
            venue_counter[v] += 1
    if venue_counter:
        top_venues = venue_counter.most_common(args.top_n)
        save_bar(
            x_labels=[k for k,_ in top_venues],
            counts=[v for _,v in top_venues],
            title=f"Top {args.top_n} Venues (by citing items)",
            xlabel="Venue", ylabel="Count",
            out_path=os.path.join(args.fig_dir, "fig7_top_venues.png"),
            rotate=60
        )
    else:
        write_txt(os.path.join(args.fig_dir, "fig7_top_venues.txt"),
                  "No venue_name_openalex data available.")

    # 8) Hit flags summary
    hit_keys = [
        "hit_keyphrase","hit_odp_url","hit_named_odp","hit_reuse_near_pattern",
        "hit_wop","hit_neg_software","hit_pattern_without_ontology","title_and_abstract_rule"
    ]
    hits_counter = Counter()
    total_rows = len(all_rows)
    for r in all_rows:
        for k in hit_keys:
            val = str(r.get(k) or "").strip()
            if val in ("1", "True", "true"):
                hits_counter[k] += 1
    if total_rows > 0:
        save_bar(
            x_labels=hit_keys,
            counts=[hits_counter.get(k, 0) for k in hit_keys],
            title="Signal Hits Across All Citing Items",
            xlabel="Signal", ylabel="Count",
            out_path=os.path.join(args.fig_dir, "fig8_signal_hits.png"),
            rotate=30
        )
        # Save rates as text
        with open(os.path.join(args.fig_dir, "signal_hit_rates.txt"), "w", encoding="utf-8") as f:
            for k in hit_keys:
                cnt = hits_counter.get(k, 0)
                rate = (cnt / total_rows * 100.0) if total_rows else 0.0
                f.write(f"{k}: {cnt} ({rate:.1f}%)\n")
    else:
        write_txt(os.path.join(args.fig_dir, "fig8_signal_hits.txt"),
                  "No rows to summarize hit flags.")

    # 9) Label distribution by seed year (infer from folder name)
    seed_year_counts = defaultdict(Counter)  # seed_year -> label -> count
    for f in classified_files:
        parts = os.path.normpath(f).split(os.sep)
        # expect data/<year>/<file>-classified.csv
        seed_year = None
        for p in parts:
            if p.isdigit() and len(p) == 4:
                try:
                    seed_year = int(p)
                    break
                except Exception:
                    pass
        rows, _ = read_classified_rows(f)
        if seed_year is None or not rows:
            continue
        for r in rows:
            lab = (r.get("odpreuse_label") or "").strip()
            if lab:
                seed_year_counts[seed_year][lab] += 1
    if seed_year_counts:
        cats = sorted(seed_year_counts.keys())
        labs = ["Likely", "Possible", "Unlikely"]
        stacks = { lab: [seed_year_counts[y][lab] for y in cats] for lab in labs }
        save_stacked_bars(
            categories=cats, stacks_dict=stacks,
            title="Label Distribution by SEED Paper Year",
            xlabel="Seed Paper Year", ylabel="Citing Items",
            out_path=os.path.join(args.fig_dir, "fig9_labels_by_seed_year.png"),
            rotate=0
        )
    else:
        write_txt(os.path.join(args.fig_dir, "fig9_labels_by_seed_year.txt"),
                  "Could not infer seed years from folder names under data/.")

    # 10) Save an overview CSV with high-level counts
    overview_csv = os.path.join(args.fig_dir, "stats_overview.csv")
    with open(overview_csv, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["metric","value"])
        w.writerow(["n_classified_files", len(classified_files)])
        w.writerow(["n_total_citing_rows", len(all_rows)])
        for lab in ["Likely","Possible","Unlikely"]:
            w.writerow([f"count_{lab.lower()}", label_counts.get(lab, 0)])

    # README
    readme_path = os.path.join(args.fig_dir, "_README.txt")
    write_txt(readme_path, (
        "Figures generated by make_odp_figures.py\n"
        "- fig1_labels_pie.png: Overall label distribution.\n"
        "- fig2_likely_time_series.png: Likely cases over citing years.\n"
        "- fig3_top_odp_names.png: Most frequent ODP names matched.\n"
        "- fig4_abstract_availability.png: Share of items with abstract fetched.\n"
        "- fig5_labels_by_year.png: Stacked labels by citing year.\n"
        "- fig6_hist_citing_counts.png + stats_citing_counts.txt: Citing-counts per seed (needs summary file).\n"
        "- fig7_top_venues.png: Most frequent venues.\n"
        "- fig8_signal_hits.png + signal_hit_rates.txt: Count and rate of rule hits.\n"
        "- fig9_labels_by_seed_year.png: Labels grouped by seed paper year (from folder names).\n"
        "- stats_overview.csv: Global counts.\n"
        "If a figure cannot be created due to missing data, a .txt is saved explaining why.\n"
    ))
    print(f"Saved figures to: {os.path.abspath(args.fig_dir)}")

if __name__ == "__main__":
    main()
