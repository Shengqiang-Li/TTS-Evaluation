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

import parselmouth
import numpy as np


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_f0_features_using_parselmouth(audio, cfg, speed=1):
    """Using parselmouth to extract the f0 feature.
    Args:
        audio
        mel_len
        hop_length
        fs
        f0_min
        f0_max
        speed(default=1)
    Returns:
        f0: numpy array of shape (frame_len,)
        pitch_coarse: numpy array of shape (frame_len,)
    """
    hop_size = int(np.round(cfg.hop_size * speed))

    # Calculate the time step for pitch extraction
    time_step = hop_size / cfg.sample_rate * 1000

    f0 = (
        parselmouth.Sound(audio, cfg.sample_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=cfg.f0_min,
            pitch_ceiling=cfg.f0_max,
        )
        .selected_array["frequency"]
    )
    return f0


def get_cents(f0_hz):
    """
    F_{cent} = 1200 * log2 (F/440)

    Reference:
        APSIPA'17, Perceptual Evaluation of Singing Quality
    """
    voiced_f0 = f0_hz[f0_hz != 0]
    return 1200 * np.log2(voiced_f0 / 440)


def get_pitch_sub_median(f0_hz):
    """
    f0_hz: (,T)
    """
    f0_cent = get_cents(f0_hz)
    return f0_cent - np.median(f0_cent)
