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
from pydub import AudioSegment


def torch_rms_norm(wav, db_level=-27.0):
    r = 10 ** (db_level / 20)
    a = torch.sqrt((wav.size(-1) * (r**2)) / torch.sum(wav**2))
    return wav * a


def extract_utmos(ref_wav, deg_wav, device):
    ref_dBFS = AudioSegment.from_file(ref_wav).dBFS
    # uses UTMOS (https://arxiv.org/abs/2204.02152) Open source (https://github.com/tarepan/SpeechMOS) following https://arxiv.org/abs/2311.12454
    mos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    audio, sr = librosa.load(deg_wav, sr=None, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0)
    # RMS norm based on the reference audio dBFS it make all models output in the same db level and it avoid issues
    audio = torch_rms_norm(audio, db_level=ref_dBFS)
    # predict UTMOS
    score = mos_predictor(audio.to(device), sr).item()
    return score
