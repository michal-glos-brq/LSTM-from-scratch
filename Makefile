# Simple makefile capable of clearing python cache and compressing the project into a zipfile
.PHONY: rmcpc

rmcpc:
	@echo "Clearing pycache ..."
	@find . -type d -name '__pycache__' -exec rm -r {} +

zip: rmcpc
	@echo zipping src
	@zip -r xglosm01.zip src Makefile requirements.txt