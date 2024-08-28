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


class SVPipeline:
    
    def __init__(self, model='eres2net', lang='zh', device='cuda'):
        self.model = model
        if self.model == 'wavlm':
            try:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "microsoft/wavlm-base-plus-sv"
                )
                self.sv_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
            except:
                self.feature_extractor  = Wav2Vec2FeatureExtractor.from_pretrained(
                    "pretrained/wavlm", sampling_rate=16000
                )
                self.sv_model = WavLMForXVector.from_pretrained("pretrained/wavlm")
            self.sv_model = self.sv_model.to(device)
        if self.model == 'eres2net':
            if lang == 'zh':
                self.sv_model = pipeline(
                    task='speaker-verification',
                    model='damo/speech_eres2net_large_200k_sv_zh-cn_16k-common',
                    model_revision='v1.0.0'
                )
            if lang == 'en':
                self.sv_model = pipeline(
                    task='speaker-verification',
                    model='iic/speech_eres2net_large_sv_en_voxceleb_16k'
                )

    def compute_cos_sim_score(self, wav_1, wav_2):
        spk1_wav, _ = librosa.load(wav_1, sr=16000)
        spk2_wav, _ = librosa.load(wav_2, sr=16000)

        if self.model == 'eres2net':
            cos_sim = self.sv_model([spk1_wav, spk2_wav])['score']
        if self.model == 'wavlm':
            inputs_1 = self.feature_extractor(
                [spk1_wav], padding=True, return_tensors="pt", sampling_rate=16000
            )
            if torch.cuda.is_available():
                for key in inputs_1.keys():
                    inputs_1[key] = inputs_1[key].to(self.sv_model.device)
            with torch.no_grad():
                embds_1 = self.sv_model(**inputs_1).embeddings
                embds_1 = embds_1[0]

            inputs_2 = self.feature_extractor(
                [wav_2], padding=True, return_tensors="pt", sampling_rate=16000
            )
            if torch.cuda.is_available():
                for key in inputs_2.keys():
                    inputs_2[key] = inputs_2[key].to(self.sv_model.device)

            with torch.no_grad():
                embds_2 = self.sv_model(**inputs_2).embeddings
                embds_2 = embds_2[0]
            cos_sim = F.cosine_similarity(embds_1, embds_2, dim=-1).detach().cpu().numpy()
        return cos_sim
