.PHONY: help
help:
	@echo Usage: make [options] [target] ...
	@echo Valid targets:
	@echo     lint - PyLint
	@echo     test - PyTest
	@echo     prof - PyTest with profile report
	@echo     cov  - PyTest with HTML coverage report

.PHONY: lint
lint:
	@pylint src/seqlogic tests

.PHONY: test
test:
	@pytest --doctest-modules

.PHONY: prof
prof:
	@pytest --doctest-modules --profile

.PHONY: cov
cov:
	@pytest --doctest-modules --cov=src/seqlogic --cov-report=html
