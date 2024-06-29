# Copyright (c) 2024, Shengqiang Li (shengqiangli@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="compute evaluation metrics for tts model.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="json file recording sythesised wav and evaluation metrics",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="result file recoring the averged metrics",
    )
    return parser.parse_args()


def main():
    args = get_args()
    pesq_lst = []
    wer_lst = []
    f0_rmse_lst = []
    utmos_lst = []
    mcd_lst = []
    with open(args.input_file, 'r') as fin:
        for line in fin.readlines():
            line_dict = json.loads(line)
            pesq = float(line_dict['pesq'])
            wer = float(line_dict['wer'])
            f0_rmse = float(line_dict['f0_rmse'])
            utmos = float(line_dict['utmos'])
            mcd = float(line_dict['mcd'])

            pesq_lst.append(pesq)
            wer_lst.append(wer)
            f0_rmse_lst.append(f0_rmse)
            utmos_lst.append(utmos)
            mcd_lst.append(mcd)
    out_dict = {}
    out_dict['pesq'] = str(sum(pesq_lst) / len(pesq_lst))
    out_dict['wer'] = str(sum(wer_lst) / len(wer_lst))
    out_dict['f0_rmse'] = str(sum(f0_rmse_lst) / len(f0_rmse_lst))
    out_dict['utmos'] = str(sum(utmos_lst) / len(utmos_lst))
    out_dict['mcd'] = str(sum(mcd_lst) / len(mcd_lst))
    with open(args.result_file, 'w') as fout:
        json.dump(out_dict, fout)
    print(out_dict)
    print(f'Save result to {args.result_file}')


if __name__ == "__main__":
    main()
