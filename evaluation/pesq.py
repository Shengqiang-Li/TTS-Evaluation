## Copyright (c) 2024, Shengqiang Li (shengqiangli@mail.nwpu.edu.cn)
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

import librosa
import numpy as np
from pypesq import pesq


def extract_pesq(audio_ref, audio_deg, method):
    # Load audio
    audio_ref, ref_sr = librosa.load(audio_ref)
    audio_deg, deg_sr = librosa.load(audio_deg)

    # Resample
    if ref_sr != 16000:
        audio_ref = librosa.resample(audio_ref, orig_sr=ref_sr, target_sr=16000)
    if deg_sr != 16000:
        audio_deg = librosa.resample(audio_deg, orig_sr=deg_sr, target_sr=16000)
    fs = 16000

    # Audio length alignment
    if len(audio_ref) != len(audio_deg):
        if method == "cut":
            length = min(len(audio_ref), len(audio_deg))
            audio_ref = audio_ref[:length]
            audio_deg = audio_deg[:length]
        elif method == "dtw":
            _, wp = librosa.sequence.dtw(audio_ref, audio_deg, backtrack=True)
            audio_ref_new = []
            audio_deg_new = []
            for i in range(wp.shape[0]):
                ref_index = wp[i][0]
                deg_index = wp[i][1]
                audio_ref_new.append(audio_ref[ref_index])
                audio_deg_new.append(audio_deg[deg_index])
            audio_ref = np.array(audio_ref_new)
            audio_deg = np.array(audio_deg_new)
            assert len(audio_ref) == len(audio_deg)

    # Compute pesq
    score = pesq(audio_ref, audio_deg, fs)
    return score
