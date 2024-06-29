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

from pymcd.mcd import Calculate_MCD


def extract_mcd(audio_ref, audio_deg):
    """Extract Mel-Cepstral Distance for a two given audio.
    Args:
        audio_ref: The given reference audio. It is an audio path.
        audio_deg: The given synthesized audio. It is an audio path.
    """

    mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
    mcd_value = mcd_toolbox.calculate_mcd(audio_ref, audio_deg)
    return mcd_value
