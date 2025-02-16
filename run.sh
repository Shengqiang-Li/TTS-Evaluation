#!/usr/bin/env bash

python main.py \
  --input_file examples/input.json \
  --result_file examples/result.json \
  --wav_dir examples/gen_wav \
  --lang zh \
  --method cut \
  --sim_model eres2net \
  --device 'cpu'


python average.py \
  --input_file examples/result.json \
  --result_file examples/merged.json
