#!/usr/bin/env bash

method='cut'
lang=zh
input_file=examples/input.json
gen_wav_dir=examples/gen_wav
result_file=examples/result_detail.json

python main.py \
  --input_file $input_file \
  --result_file $result_file \
  --wav_dir $gen_wav_dir \
  --lang $lang \
  --method $method \
  --sim_model eres2net \
  --device cpu &


python average.py \
  --input_file $result_file \
  --result_file /Users/lichengqiang/Documents/code/TTS-Evaluation/examples/result_avg.json

