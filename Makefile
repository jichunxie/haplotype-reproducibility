PYTHON ?= python

.PHONY: verify figures test clean

verify:
	$(PYTHON) scripts/verify_checksums.py
	$(PYTHON) scripts/audit_results.py
	PYTHONPATH=analysis/estimation:analysis/simulation $(PYTHON) analysis/simulation/test_ranking_oracle_v76.py
	PYTHONPATH=analysis/estimation:analysis/simulation $(PYTHON) analysis/simulation/test_ranking_frequency_design_v77.py

figures:
	MPLCONFIGDIR=.mplconfig $(PYTHON) scripts/build_paper_outputs.py

test: verify figures

clean:
	$(PYTHON) scripts/clean_generated.py
