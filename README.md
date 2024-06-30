# TTS-Evaluation
This repository gathers our efforts to evaluate/compare the current Multi-speaker TTS systems using objective metrics.
# Metrics
## UTMOS
Following [HierSpeech++ paper](https://arxiv.org/abs/2311.12454), we have used the [UTMOS model](https://arxiv.org/abs/2204.02152) to predict the Naturalness Mean Opinion Score (nMOS). In the HierSpeech++ paper, the authors have used the open-source version of UTMOS\footnote{https://github.com/tarepan/SpeechMOS}, and the presented results of human nMOS and UTMOS are almost aligned. 
Although this can not be considered an absolute evaluation metric, it can be used to easily compare models in quality terms. 
## WER/CER
Following previous works, we evaluate pronunciation accuracy using an ASR model. For it, we have used the [Whisper Large v3 model](https://huggingface.co/openai/whisper-large-v3) for english text and the [Paraformer model](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) for chinese text. Additionally, we also removed all text punctuation before computing WER/CER.
## Cosine Similarity
To compare the similarity between the synthesized voice and the original speaker, we compute the [Speaker Encoder Cosine Similarity (SECS)](https://arxiv.org/abs/2104.05557).
We have two choices to compute the SECS, which contains [ERes2Net-large](https://modelscope.cn/models/iic/speech_eres2net_large_200k_sv_zh-cn_16k-common) and [WavLM-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv).
## Mel Cepstral Distortion
We compute the mel cepstral distortion between the predicted wav and the ground-truth wav as follows,
$$
\operatorname{MCD}\left(\mathbf{c}_p, \mathbf{c}_g\right)=\frac{10}{\ln 10} \sqrt{2 \sum_{k=1}^{M_c}\left[c_p(k)-c_g(k)\right]^2}
$$
where $c_p$ and $c_g$ are the predicted and ground-truth Mel
cepstrum coefficient vectors respectively, and $M_c$ refers
to the dimensionality of Mel cepstrum coefficients.
## PESQ
We compute the perceptual evaluation of speech quality score via pypesq.
## FO RMSE
We compute the root mean root mean square error in F0 estimation as follows,
$$
\operatorname{RMSE}(\mathbf{f0}, \hat{\mathbf{f0}})=\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{f0}_{i}-\hat{\mathbf{f0}}_{i}\right)^2}
$$
where $\mathbf{f0}$ refers to the ground-truth f0, and $\hat{\mathbf{f0}}$ refers to the predicted f0. $N$ refers to the dimension of the vector.
# Installation
```
pip install -r requirements
```
# Usage
```
bash run.sh
```
# Todo List
- [ ] Visqol score.
- [ ] voice/unvoice errors.
# Acknowledgement
- We borrow a little of code from [Amphion](https://github.com/open-mmlab/Amphion) for some evaluation mectrics computation.