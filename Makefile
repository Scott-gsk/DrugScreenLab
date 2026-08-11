.PHONY: test validate
test:
	python -m pytest
validate:
	python -m drug_screen.data.registry --root data
