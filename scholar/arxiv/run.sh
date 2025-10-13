python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python odp_paper_harvester.py \
  --query "ontology design pattern" \
  --query "ontology pattern reuse" \
  --email your.name@kit.edu \
  --outdir ./odp_results \
  --filter_big_publishers \
  --attempt_institutional_pdf 
