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


def clean_text_en(text):
    punctuation_str = string.punctuation
    for i in punctuation_str:
        text = text.replace(i, '')
    text = text.lower()
    return text


def clean_text_zh(text):
    punctuation_str = hanzi.punctuation
    for i in punctuation_str:
        text = text.replace(i, '')
    return text


def infer_en(audio_deg):
    model = WhisperModel(WhisperModel("large-v3"))
    segments, info = model.transcribe(
        audio_deg,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700)
    )
    content_pred = ""
    for segment in segments:
        content_pred += segment.text
    content_pred = content_pred.rstrip().strip()
    return content_pred


def infer_zh(audio_deg):
    asr_repo_dir = snapshot_download("pengzhendong/offline-paraformer-zh")
    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=f"{asr_repo_dir}/model.onnx",
        tokens=f"{asr_repo_dir}/tokens.txt",
        num_threads=1,
        sample_rate=16000,
        feature_dim=80
    )
    audio, _ = librosa.load(audio_deg, sr=16000)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)
    recognizer.decode_stream(stream)
    rec_text = stream.result.text
    return rec_text


def get_wer(ref_text, hyp_text, lang):
    if lang == 'en':
        ref_text = clean_text_en(ref_text)
        hyp_text = clean_text_en(hyp_text)
    elif lang == 'zh':
        ref_text = clean_text_zh(ref_text)
        hyp_text = clean_text_zh(hyp_text)
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
