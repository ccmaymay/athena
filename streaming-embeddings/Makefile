INPUT_PATH ?= ../word2vec/word2vec/trunk/text8
OUTPUT_DIR ?= /export/projects/cmay
OUTPUT_STEM ?= text8
ROOT_DIR ?= ..


SUFFIXES := 1-100.1-100 1-100.401-500 1-100.1601-1700 1-100.6401-6500 \
    401-500.401-500 401-500.1601-1700 401-500.6401-6500 \
    1601-1700.1601-1700 1601-1700.6401-6500 \
    6401-6500.6401-6500


.PHONY: all
all: $(patsubst %,$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.ssw2v.png,$(SUFFIXES)) \
    $(patsubst %,$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.0.png,$(SUFFIXES)) \
    $(patsubst %,$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.1.png,$(SUFFIXES)) \
    $(patsubst %,$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.2.png,$(SUFFIXES)) \
    $(OUTPUT_DIR)/$(OUTPUT_STEM).counts.png \
    $(OUTPUT_DIR)/$(OUTPUT_STEM).counts.filter-missing.png

.SECONDARY:

$(OUTPUT_DIR)/$(OUTPUT_STEM).lm: $(INPUT_PATH)
	$(ROOT_DIR)/build/lib/naive-lm-train-raw $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).sslm: $(INPUT_PATH)
	$(ROOT_DIR)/build/lib/spacesaving-lm-train-raw $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).lm.counts $(OUTPUT_DIR)/$(OUTPUT_STEM).sslm.counts: %.counts: %
	$(ROOT_DIR)/build/lib/lm-print --with-counts $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).1-end: $(OUTPUT_DIR)/$(OUTPUT_STEM).lm
	$(ROOT_DIR)/build/lib/lm-print $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).1-100: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-end
	head -n 100 < $^ > $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).401-500: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-end
	head -n 500 < $^ | tail -n 100 > $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-end
	head -n 1700 < $^ | tail -n 100 > $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).6401-6500: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-end
	head -n 6500 < $^ | tail -n 100 > $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).1-100.1-100: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-100
	python $(ROOT_DIR)/streaming-embeddings/unique-pairs.py $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).401-500.401-500: $(OUTPUT_DIR)/$(OUTPUT_STEM).401-500
	python $(ROOT_DIR)/streaming-embeddings/unique-pairs.py $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700.1601-1700: $(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700
	python $(ROOT_DIR)/streaming-embeddings/unique-pairs.py $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).6401-6500.6401-6500: $(OUTPUT_DIR)/$(OUTPUT_STEM).6401-6500
	python $(ROOT_DIR)/streaming-embeddings/unique-pairs.py $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).1-100.401-500: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-100 $(OUTPUT_DIR)/$(OUTPUT_STEM).401-500
	bash $(ROOT_DIR)/streaming-embeddings/pairs.bash $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).1-100.1601-1700: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-100 $(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700
	bash $(ROOT_DIR)/streaming-embeddings/pairs.bash $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).1-100.6401-6500: $(OUTPUT_DIR)/$(OUTPUT_STEM).1-100 $(OUTPUT_DIR)/$(OUTPUT_STEM).6401-6500
	bash $(ROOT_DIR)/streaming-embeddings/pairs.bash $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).401-500.1601-1700: $(OUTPUT_DIR)/$(OUTPUT_STEM).401-500 $(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700
	bash $(ROOT_DIR)/streaming-embeddings/pairs.bash $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).401-500.6401-6500: $(OUTPUT_DIR)/$(OUTPUT_STEM).401-500 $(OUTPUT_DIR)/$(OUTPUT_STEM).6401-6500
	bash $(ROOT_DIR)/streaming-embeddings/pairs.bash $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700.6401-6500: $(OUTPUT_DIR)/$(OUTPUT_STEM).1601-1700 $(OUTPUT_DIR)/$(OUTPUT_STEM).6401-6500
	bash $(ROOT_DIR)/streaming-embeddings/pairs.bash $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).w2v $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v.0 $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v.1 $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v.2: $(INPUT_PATH)
	$(ROOT_DIR)/build/lib/word2vec-train-raw $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).ssw2v: %: $(INPUT_PATH)
	$(ROOT_DIR)/build/lib/spacesaving-word2vec-train-raw $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v: $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v $(OUTPUT_DIR)/$(OUTPUT_STEM).%
	$(ROOT_DIR)/build/lib/sgns-model-print-similarity $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.ssw2v: $(OUTPUT_DIR)/$(OUTPUT_STEM).ssw2v $(OUTPUT_DIR)/$(OUTPUT_STEM).%
	$(ROOT_DIR)/build/lib/sgns-model-print-similarity $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.0: $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v.0 $(OUTPUT_DIR)/$(OUTPUT_STEM).%
	$(ROOT_DIR)/build/lib/sgns-model-print-similarity $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.1: $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v.1 $(OUTPUT_DIR)/$(OUTPUT_STEM).%
	$(ROOT_DIR)/build/lib/sgns-model-print-similarity $^ $@
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.2: $(OUTPUT_DIR)/$(OUTPUT_STEM).w2v.2 $(OUTPUT_DIR)/$(OUTPUT_STEM).%
	$(ROOT_DIR)/build/lib/sgns-model-print-similarity $^ $@

$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.ssw2v.png: $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.ssw2v
	python $(ROOT_DIR)/streaming-embeddings/plot-sim.py --sim-name spacesaving-word2vec --baseline-sim-name word2vec --output-path $@ $^
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.0.png: $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.0
	python $(ROOT_DIR)/streaming-embeddings/plot-sim.py --sim-name word2vec.0 --baseline-sim-name word2vec --output-path $@ $^
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.1.png: $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.1
	python $(ROOT_DIR)/streaming-embeddings/plot-sim.py --sim-name word2vec.1 --baseline-sim-name word2vec --output-path $@ $^
$(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.2.png: $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v $(OUTPUT_DIR)/$(OUTPUT_STEM).%.sim.w2v.2
	python $(ROOT_DIR)/streaming-embeddings/plot-sim.py --sim-name word2vec.2 --baseline-sim-name word2vec --output-path $@ $^

$(OUTPUT_DIR)/$(OUTPUT_STEM).counts.png: $(OUTPUT_DIR)/$(OUTPUT_STEM).lm.counts $(OUTPUT_DIR)/$(OUTPUT_STEM).sslm.counts
	python $(ROOT_DIR)/streaming-embeddings/plot-counts.py --title 'spacesaving counts' --output-path $@ $^

$(OUTPUT_DIR)/$(OUTPUT_STEM).counts.filter-missing.png: $(OUTPUT_DIR)/$(OUTPUT_STEM).lm.counts $(OUTPUT_DIR)/$(OUTPUT_STEM).sslm.counts
	python $(ROOT_DIR)/streaming-embeddings/plot-counts.py --title 'spacesaving counts (filtered)' --filter-missing --output-path $@ $^
