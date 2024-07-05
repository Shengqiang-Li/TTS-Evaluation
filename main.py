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
from tqdm import tqdm
from evaluation.pesq import extract_pesq
from evaluation.f0_rmse import extract_f0rmse
from evaluation.similarity import extract_sim_eres2net, extract_sim_wavlm_base
from evaluation.wer import infer_zh, infer_en, get_wer
from evaluation.mel_cepstral_distortion import extract_mcd
from evaluation.utmos import extract_utmos


def get_args():
    parser = argparse.ArgumentParser(
        description="compute evaluation metrics for tts model.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="json file contains reference wav, sythesised wav, text.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="directory of synthesised wav"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="result file recoring the metrics",
    )
    parser.add_argument(
        "--method",
        type=str,
        default='cut',
        help="choose between cut and dtw.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="language of the text, choose between zh and en"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="choose cuda"
    )
    parser.add_argument(
        "--sim_model",
        type=str,
        default='eres2net',
        help="choose between eres2net and wavlm"
    )
    return parser.parse_args()


def main():
    args = get_args()
    assert args.lang in ['zh', 'en']
    assert args.method in ['cut', 'dtw']
    assert args.sim_model in ['eres2net', 'wavlm']
    with open(args.input_file, 'r') as fin:
        with open(args.result_file, 'w') as fout:
            for line in tqdm(fin.readlines()):
                line_dict = json.loads(line)
                ref_text = line_dict['text']
                audio_ref = line_dict['gt_wav']
                wav_name = line_dict['out_key'] + '.wav'
                audio_deg = f'{args.wav_dir}/{wav_name}'
                result_dict = {'gen_wav': audio_deg}

                pesq = extract_pesq(audio_ref, audio_deg, method=args.method)
                result_dict['pesq'] = pesq

                if args.sim_model == 'wavlm':
                    cos_sim = extract_sim_wavlm_base(audio_ref, audio_deg, args.device)
                elif args.sim_model == 'eres2net':
                    cos_sim = extract_sim_eres2net(audio_ref, audio_deg, args.lang)['score']
                result_dict['cos_sim'] = cos_sim

                f0_rmse = extract_f0rmse(audio_ref, audio_deg, method=args.method)
                result_dict['f0_rmse'] = f0_rmse

                if args.lang == 'zh':
                    hyp_text = infer_zh(audio_deg)
                    wer_ = get_wer(ref_text, hyp_text, 'zh')
                elif args.lang == 'en':
                    hyp_text = infer_en(audio_deg)
                    wer_ = get_wer(ref_text, hyp_text, 'en')
                result_dict['wer'] = wer_['wer']
                result_dict['ref'] = wer_['ref']
                result_dict['hyp'] = wer_['hyp']
                result_dict['del'] = wer_['del']
                result_dict['sub'] = wer_['sub']
                result_dict['ins'] = wer_['ins']

                mcd = extract_mcd(audio_ref, audio_deg)
                result_dict['mcd'] = mcd

                utmos = extract_utmos(audio_ref, audio_deg, args.device)
                result_dict['utmos'] = utmos
                fout.writelines(json.dumps(result_dict, ensure_ascii=False) + '\n')
        return


if __name__ == "__main__":

    main()
