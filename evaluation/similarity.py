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

import torch
import librosa
import torch.nn.functional as F
from modelscope.pipelines import pipeline
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def extract_sim_eres2net(wav1_path, wav_2path, lang):
    assert lang in ['zh', 'en']
    spk1_wav, _ = librosa.load(wav1_path, sr=16000)
    spk2_wav, _ = librosa.load(wav_2path, sr=16000)
    if lang == 'zh':
        sv_pipline = pipeline(
            task='speaker-verification',
            model='damo/speech_eres2net_large_200k_sv_zh-cn_16k-common',
            model_revision='v1.0.0'
        )
    if lang == 'en':
        sv_pipline = pipeline(
            task='speaker-verification',
            model='iic/speech_eres2net_large_sv_en_voxceleb_16k'
        )
    # 不同说话人语音
    result = sv_pipline([spk1_wav, spk2_wav])
    return result


def extract_sim_wavlm_base(wav_1, wav_2, device):
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    except:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "pretrained/wavlm", sampling_rate=16000
        )
        model = WavLMForXVector.from_pretrained("pretrained/wavlm")
    model = model.to(device)

    wav_1, _ = librosa.load(wav_1, sr=16000)
    inputs_1 = feature_extractor(
        [wav_1], padding=True, return_tensors="pt", sampling_rate=16000
    )
    if torch.cuda.is_available():
        for key in inputs_1.keys():
            inputs_1[key] = inputs_1[key].to(device)
    with torch.no_grad():
        embds_1 = model(**inputs_1).embeddings
        embds_1 = embds_1[0]

    wav_2, _ = librosa.load(wav_2, sr=16000)

    inputs_2 = feature_extractor(
        [wav_2], padding=True, return_tensors="pt", sampling_rate=16000
    )
    if torch.cuda.is_available():
        for key in inputs_2.keys():
            inputs_2[key] = inputs_2[key].to(device)

    with torch.no_grad():
        embds_2 = model(**inputs_2).embeddings
        embds_2 = embds_2[0]
    sim = F.cosine_similarity(embds_1, embds_2, dim=-1).detach().cpu().numpy()
    return sim
