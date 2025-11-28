# ODP Reuse Benchmark: Harvesting, Classifying, and Verifying Ontology Design Pattern Reuse

This repository provides a reproducible pipeline to (i) **collect papers about Ontology Design Patterns (ODPs)**, (ii) **identify ontologies that cite and/or reuse ODPs**, and (iii) **quantify and validate reuse** across multiple sources (Google, GitHub, OntoPortal instances, and SPARQL endpoints). The final products are (a) a **statistics report** for ODP reuse, and (b) a **gold-standard dataset** of ontologies that reuse **100%** of the patterns detected for them.

> **Paper link:** <a href="https://doi.org/" target="_blank">DOI</a>


---

## Contents

- `scholar/` – paper-centric pipeline
  - `arxiv/odp_paper_harvester.py` – harvests ODP-related papers from **arXiv, OpenAlex, Crossref, and Unpaywall** (PDF retrieval best-effort).
  - `paper_citations_openalex.py` – queries OpenAlex for per-paper citation counts/metadata.
  - `classify_odp_reuse.py` – rule-based screening of citing papers for likely ODP reuse.
  - `odp_summary_and_plots.py`, `odp_figures.py` – produces summary tables and figures under `scholar/output/`.
  - `arxiv/run.sh` – minimal example (creates venv, installs `arxiv/requirements.txt`, runs harvester).
- `MultiSource/` – ontology-centric pipeline
  - `odp_multi_source_finder.py` – discovers ontologies reusing ODPs by searching **Google Custom Search**, **GitHub Code Search**, **BioPortal**, **MatPortal**, **LOV**, and optionally **SPARQL** endpoints. It can auto-discover pattern IRIs from the official `odpa/patterns-repository`.
  - `build_github_dataset.py` – builds a local dataset from a GitHub reuse CSV, cloning candidate repos and structuring them on disk.
  - `odp_stats.py` – computes reuse statistics from local repositories/ontologies and generates plots.
  - `gold_dataset.py` – materializes a **gold-standard dataset** filtered to ontologies that reuse **100%** of their detected patterns; supports multiple similarity backends.
  - `run_multiple_source.sh` – example orchestration script; sources `.env`, clones the pattern repo, runs statistics, and builds gold datasets.
- `scholar/output/` – default output location for the paper pipeline.
- `.env.example` – template for API keys used by the multi-source search (see _Configuration_).

---

## Data Sources

- **Papers:** arXiv, OpenAlex, Crossref, Unpaywall (PDF retrieval when possible).
- **Ontology reuse (web/code):** Google Custom Search (programmable search), GitHub Code Search.
- **Ontology portals:** BioPortal and MatPortal (OntoPortal stack).
- **Linked data catalogs:** LOV API and/or SPARQL endpoint(s).

---

## Installation

We recommend Python **3.10–3.11** and a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A consolidated `requirements.txt` is provided in this folder (baseline derived from imports). The submodule `scholar/arxiv/requirements.txt` is also kept for the minimal arXiv harvester demo.

---

## Configuration

Create a `.env` file at the project root with the following variables (copy from `.env.example`):

```dotenv
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
GITHUB_TOKEN=...
BIOPORTAL_API_KEY=...
MATPORTAL_API_KEY=...
```

---

## Reproducible Pipelines

### A) Harvest ODP Papers (scholar/arxiv)

1. Create and activate a fresh environment (see _Installation_).  
2. Install arXiv harvester deps (already covered by top-level `requirements.txt`; alternatively: `pip install -r scholar/arxiv/requirements.txt`).  
3. Run the harvester with explicit queries and a contact email:

```bash
cd scholar/arxiv
python odp_paper_harvester.py   --query "ontology design pattern"   --query "ontology pattern reuse"   --email your.name@example.edu   --outdir ../output/odp_results   --filter_big_publishers   --attempt_institutional_pdf
```

Outputs (under `scholar/output/` by default) include metadata tables and any retrieved PDFs.

Optionally augment with OpenAlex citations:

```bash
cd ..
python paper_citations_openalex.py --input ./output/odp_results/metadata.csv --out ./output/citations.csv
```

Classify likely ODP reuse in citing papers:

```bash
python classify_odp_reuse.py   --input ./output/citations.csv   --pattern-library-dir ../MultiSource/patterns   --log-dir ./output/logs   --out ./output/classified.csv
```

Produce summaries and figures:

```bash
python odp_summary_and_plots.py --input ./output/classified.csv --outdir ./output/wop
python odp_figures.py --input ./output/classified.csv --outdir ./output/wop/figures
```

---

### B) Discover Ontology Reuse from Multiple Sources (MultiSource)

Prepare credentials in `.env` (see _Configuration_). Then:

```bash
cd MultiSource
bash run_multiple_source.sh
```

Or run the main components explicitly.

**1) Multi-source finder:** harvest candidate ontologies reusing ODPs by IRI.

```bash
python odp_multi_source_finder.py   --patterns-repo ./patterns-repository \ 
  --sources github,google,bioportal,matportal,lov   --query-limit 200   --out report/
```

**2) Build a local GitHub dataset** from the finder’s CSV (clones repos and organizes files).

```bash
python build_github_dataset.py   --csv report/github.csv   --patterns-repo ./patterns-repository   --out-dir github_dataset
```

**3) Compute reuse statistics** (character/word/SBERT/backends; plots saved to output directory).

```bash
python odp_stats.py   --base github_dataset   --out-dir github_dataset_stats   --all-backends
```

**4) Materialize the gold-standard dataset** (ontologies that reuse **100%** of their detected patterns).

```bash
# Pick a backend: hybrid | char | word | sbert | exact
python gold_dataset.py   --base github_dataset   --out-dir gold_github   --sim-backend hybrid   --materialize symlink   --dataset-percents "100"   --make-graph-100
```

Repeat the same for any additional base directory (e.g., `Ontologies_reusingODPs`).

---

## Citation

If you use this code or the datasets, please cite the accompanying paper:  
**“ODP Reuse Benchmark …”** <a href="https://doi.org/" target="_blank">DOI</a>


---

## License

This project is released under the [MIT License](LICENSE).

