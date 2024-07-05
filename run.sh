#!/usr/bin/env bash

python main.py \
  --input_file examples/input.json \
  --result_file examples/result_detail.json \
  --wav_dir examples/gen_wav \
  --lang zh \
  --method cut \
  --sim_model eres2net \
  --device 'cuda:0'


python average.py \
  --input_file examples/result_detail.json \
  --result_file examples/result_avg.json

