.PHONY: help
help:
	@echo Usage: make [options] [target] ...
	@echo Valid targets:
	@echo     lint  - PyLint
	@echo     test  - PyTest
	@echo     prof  - PyTest with profile report
	@echo     cov   - PyTest with HTML coverage report
	@echo     build - Build source/binary distributions

PYTHON := python
PYLINT := pylint
PYTEST := pytest

.PHONY: lint
lint:
	@$(PYLINT) src/seqlogic tests

.PHONY: test
test:
	@$(PYTEST) --doctest-modules

.PHONY: prof
prof:
	@$(PYTEST) --doctest-modules --profile

.PHONY: cov
cov:
	@$(PYTEST) --doctest-modules --cov=src/seqlogic --cov-report=html

.PHONY: build
build:
	@$(PYTHON) -m build
