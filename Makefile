
.PHONY: rmcpc

rmcpc:
	@echo "Clearing pycache ..."
	@find . -type d -name '__pycache__' -exec rm -r {} +

zip: rmcpc
	@echo zipping src
	@zip -r src.zip src