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

import math
import torch
import librosa
import numpy as np
from evaluation.utils import (
    JsonHParams,
    get_f0_features_using_parselmouth,
    get_pitch_sub_median)

ZERO = 1e-8


def extract_f0rmse(
    audio_ref,
    audio_deg,
    hop_length=256,
    f0_min=50,
    f0_max=1100,
    method='cut',
    need_mean=True,
):
    """Compute F0 Root Mean Square Error (RMSE) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    # Load audio
    audio_ref, ref_sr = librosa.load(audio_ref)
    audio_deg, deg_sr = librosa.load(audio_deg)
    fs = deg_sr
    if ref_sr != deg_sr:
        audio_ref = librosa.resample(audio_ref, orig_sr=ref_sr, target_sr=deg_sr)
    # Initialize config for f0 extraction
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = 256
    cfg.pitch_max = f0_max
    cfg.pitch_min = f0_min

    # Extract f0
    f0_ref = get_f0_features_using_parselmouth(
        audio_ref,
        cfg,
    )

    f0_deg = get_f0_features_using_parselmouth(
        audio_deg,
        cfg,
    )

    # Subtract mean value from f0
    if need_mean:
        f0_ref = torch.from_numpy(f0_ref)
        f0_deg = torch.from_numpy(f0_deg)

        f0_ref = get_pitch_sub_median(f0_ref).numpy()
        f0_deg = get_pitch_sub_median(f0_deg).numpy()

    # Avoid silence
    min_length = min(len(f0_ref), len(f0_deg))
    if min_length <= 1:
        return 0

    # F0 length alignment
    if method == "cut":
        length = min(len(f0_ref), len(f0_deg))
        f0_ref = f0_ref[:length]
        f0_deg = f0_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
        f0_gt_new = []
        f0_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            f0_gt_new.append(f0_ref[gt_index])
            f0_pred_new.append(f0_deg[pred_index])
        f0_ref = np.array(f0_gt_new)
        f0_deg = np.array(f0_pred_new)
        assert len(f0_ref) == len(f0_deg)

    # Compute RMSE
    f0_mse = np.square(np.subtract(f0_ref, f0_deg)).mean()
    f0_rmse = math.sqrt(f0_mse)

    return f0_rmse
