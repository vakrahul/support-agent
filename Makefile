.PHONY: help setup fixture test eda eda-fixture eval eval-live leakage human-pack agreement architecture clean

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
	@echo "  make human-pack   emit blank human-scoring CSVs (fill by hand, no fakes)"
	@echo "  make agreement    score filled human CSVs (fails loudly if empty)"

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

human-pack:
	$(PY) scripts/make_human_scoring_pack.py

agreement:
	$(PY) scripts/score_agreement.py

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache qdrant_storage
	rm -f data/sample/*.parquet data/sample/*.csv report/figures/*.png
