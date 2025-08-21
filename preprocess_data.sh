#!/bin/bash
python3 tools/preprocess_data.py \
--input myData/openwebtext_100k.jsonl \
--output-prefix my-gpt2 \
--vocab-file myData/gpt2-vocab.json \
--merge-file myData/gpt2-merges.txt \
--tokenizer-type GPT2BPETokenizer \
--append-eod \
--workers 4