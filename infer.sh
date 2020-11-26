#!/usr/bin/env bash

python infer-cpu.py \
    --infer_file luge-dialogue/data/test.txt \
    --output_name response \
    --model UnifiedTransformer \
    --task DialogGeneration \
    --config_path luge-dialogue/config/12L.json \
    --vocab_path luge-dialogue/config/vocab.txt \
    --spm_model_file luge-dialogue/config/spm.model \
    --do_generation true \
    --is_cn true \
    --data_format numerical \
    --file_format file \
    --init_checkpoint luge-dialogue/config/12L.finetune
