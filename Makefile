help:
	@echo make dev
dev:
	rye run ipython --config=src/sentence_feature/irun.py
