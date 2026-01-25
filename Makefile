PKG := seqlogic
PYTEST := pytest

.PHONY: help
help:
	@echo Usage: make [options] [target] ...
	@echo Valid targets:
	@echo     test  - PyTest
	@echo     prof  - PyTest with profile report
	@echo     cov   - PyTest with HTML coverage report

.PHONY: test
test:
	@$(PYTEST) --doctest-modules

.PHONY: prof
prof:
	@$(PYTEST) --doctest-modules --profile

.PHONY: cov
cov:
	@$(PYTEST) --doctest-modules --cov=src/$(PKG) --cov-branch --cov-report=html
