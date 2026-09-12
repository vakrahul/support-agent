.PHONY: help setup fixture test eda eda-fixture eval eval-live leakage human-pack agreement architecture demo-image ui demo bank-judges pair-agreement report integrations clean

PY ?= python3

help:
	@echo "  make setup        install dependencies"
	@echo "  make test         run the test suite (no data, no API key needed)"
	@echo "  make eda-fixture  smoke-test the whole pipeline on synthetic data"
	@echo "  make eda          brand selection on the real Kaggle dump"
	@echo "  make leakage      fail loudly on test contamination"
	@echo "  make eval         headline results (replay mode: cached LLM, \$$0)"
	@echo "  make eval-live    re-run against the Gemini API (needs GEMINI_API_KEY)"
	@echo "  make architecture build report/figures/architecture.png"
	@echo "  make demo-image   render white-and-white real-run demo PNG (report/figures)"
	@echo "  make integrations build report/figures/integrations.png"
	@echo "  make report       compile report/REPORT.md to report/REPORT.pdf (Edge headless)"
	@echo "  make human-pack   emit blank human-scoring CSVs (fill by hand, no fakes)"
	@echo "  make agreement    score filled human CSVs (fails loudly if empty)"
	@echo "  make bank-judges  bank cached judge verdicts into eval results (no net)"
	@echo "  make pair-agreement compute paired judge-vs-human kappa/rho"
	@echo "  make demo         run diagnostic pipeline on 4 unique customer queries"
	@echo "  make ui           real-time web UI on http://localhost:8000"

setup:
	$(PY) -m pip install -r requirements.txt

fixture:
	$(PY) -m tests.make_fixture

test: fixture
	$(PY) -m pytest tests/ -q || ( \
		$(PY) -m tests.test_metrics && \
		$(PY) -m tests.test_ingest && \
		$(PY) -m tests.test_brand_select && \
		$(PY) -m tests.test_pipeline )

eda-fixture: fixture
	$(PY) scripts/run_eda.py --fixture

eda:
	$(PY) scripts/run_eda.py

eval:
	LLM_MODE=replay $(PY) scripts/run_eval.py

eval-live:
	LLM_MODE=live $(PY) scripts/run_eval.py

leakage:
	$(PY) scripts/check_leakage.py

architecture:
	$(PY) scripts/make_architecture_png.py

integrations:
	$(PY) scripts/make_integrations_png.py

report:
	$(PY) scripts/make_pdf_report.py

human-pack:
	$(PY) scripts/make_human_scoring_pack.py

agreement:
	$(PY) scripts/score_agreement.py

bank-judges:
	$(PY) scripts/bank_judges.py

pair-agreement:
	$(PY) scripts/pair_agreement.py

demo-image:
	$(PY) scripts/make_demo_image.py

ui:
	$(PY) scripts/ui.py

demo:
	$(PY) scripts/demo.py --featured

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache qdrant_storage
	rm -f data/sample/*.parquet data/sample/*.csv report/figures/*.png
