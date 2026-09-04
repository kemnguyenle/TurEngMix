# Reproduction pipeline. `make all` goes from the raw export to the tables.
PY ?= python

.PHONY: all data test llm encoders report clean

all: data test report

data: data/benchmark.csv data/splits/train.txt

data/benchmark.csv: data/raw/annotations_raw.csv scripts/normalize_annotations.py
	$(PY) scripts/normalize_annotations.py

data/splits/train.txt: data/benchmark.csv scripts/make_splits.py
	$(PY) scripts/make_splits.py

test:
	$(PY) -m pytest -q

# Costs money and needs OPENAI_API_KEY / OPENROUTER_API_KEY.
llm: data
	for task in lid ner; do for shots in 0 3; do \
	  $(PY) scripts/run_llm.py --task $$task --model gpt-4o --shots $$shots; \
	  $(PY) scripts/run_llm.py --task $$task --model qwen/qwen3-8b --shots $$shots \
	    --provider openrouter --disable-reasoning; \
	done; done

# Needs a GPU. ~6 min per run on an A6000.
encoders: data
	for m in dbmdz/bert-base-turkish-cased FacebookAI/xlm-roberta-base VRLLab/TurkishBERTweet; do \
	  for task in lid ner; do \
	    name=$$(basename $$m)_$$task; \
	    $(PY) scripts/train_encoder.py --task $$task --model $$m \
	      --out models/$$name --seeds 1 2 3 4 --fp16; \
	    for s in 1 2 3 4; do \
	      $(PY) scripts/evaluate_encoder.py --task $$task \
	        --model-dir models/$${name}_seed$$s; \
	    done; \
	done; done

report:
	$(PY) scripts/report.py

clean:
	rm -rf results/*.txt results/*_confusion.csv .pytest_cache
