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

import string
import librosa
import sherpa_onnx
import edit_distance
from zhon import hanzi
from faster_whisper import WhisperModel
from modelscope import snapshot_download


class ASRPipeline:

    def __init__(self, lang) -> None:
        self.lang = lang
        if self.lang == 'en':
            self.asr_model = WhisperModel("large-v3")
        if self.lang == 'zh':
            asr_repo_dir = snapshot_download("pengzhendong/offline-paraformer-zh")
            self.asr_model = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=f"{asr_repo_dir}/model.onnx",
                tokens=f"{asr_repo_dir}/tokens.txt",
                num_threads=1,
                sample_rate=16000,
                feature_dim=80
            )

    def clean_text_en(self, text):
        punctuation_str = string.punctuation
        for i in punctuation_str:
            text = text.replace(i, '')
        text = text.lower()
        return text


    def clean_text_zh(self, text):
        punctuation_str = hanzi.punctuation
        for i in punctuation_str:
            text = text.replace(i, '')
        return text


    def infer_en(self, audio_deg):
        segments, info = self.asr_model.transcribe(
            audio_deg,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700)
        )
        hyp_text = ""
        for segment in segments:
            hyp_text += segment.text
        hyp_text = hyp_text.rstrip().strip()
        return hyp_text


    def infer_zh(self, audio_deg):
        audio, _ = librosa.load(audio_deg, sr=16000)
        stream = self.asr_model.create_stream()
        stream.accept_waveform(16000, audio)
        self.asr_model.decode_stream(stream)
        hyp_text = stream.result.text
        return hyp_text


    def get_wer(self, ref_text, hyp_text):
        if self.lang == 'en':
            ref_text = self.clean_text_en(ref_text)
            hyp_text = self.clean_text_en(hyp_text)
        elif self.lang == 'zh':
            ref_text = self.clean_text_zh(ref_text)
            hyp_text = self.clean_text_zh(hyp_text)
        sm = edit_distance.SequenceMatcher(ref_text, hyp_text)
        wer = sm.distance() / len(ref_text)
        cor = sm.matches()

        del_error = 0
        ins_error = 0
        sub_error = 0
        for opcode in sm.get_opcodes():
            if opcode[0] == "delete":
                del_error += 1
            elif opcode[0] == "insert":
                ins_error += 1
            elif opcode[0] == "replace":
                sub_error += 1
        return {
            "ref": ref_text,
            "hyp": hyp_text,
            "wer": wer,
            "cor": cor,
            "del": del_error,
            "ins": ins_error,
            "sub": sub_error
        }
