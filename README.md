# GPT on Wikitext2 (with W&B)

PyTorch로 구현한 GPT LLM입니다.  
Hugging Face `wikitext-2-v1` 데이터셋으로 **scratch pretraining**을 진행하고, Weights & Biases(W&B)로 학습 과정을 추적합니다.

기본적인 구조는 OpenAI의 GPT2의 구조를 기반으로 하였으며 LLAMA에서 사용된 attention, SILU 등을 업데이트 하였습니다. 

Tokenizer의 경우 OpenAI의 GPT2를 사용하였습니다. 

---

## Features

- **Custom GPT Architecture**
  - Embedding dim: 384
  - Layers: 6
  - Heads: 8
  - FFN dim: 1536
  - Causal self-attention + residual connection
- **Training Utilities**
  - Mixed Precision (AMP) 지원
  - Gradient Accumulation
  - CosineAnnealingLR 스케줄러
  - Gradient Clipping
- **Experiment Tracking**
  - W&B로 loss, perplexity, learning rate, 샘플 텍스트 로깅
- **Text Generation**
  - Top‑k sampling
  - Temperature 조절
  - Repetition penalty, EOS 토큰 처리
- **Learning Modules**
  - activation functions
    - ReLU
    - GELU
    - SwiGLU
  
  - optimizer
    - AdamW
    - learning rate : 7e-4 
 
    
---

## Project Structure

```bash
.
├── LLM-wandb-*.ipynb    # 실험 노트북 (모델/학습/생성 코드)
├── checkpoint_*.pt      # 학습된 모델 체크포인트
└── README.md

```

노트북 안에는 다음 컴포넌트가 포함됩니다.

GPTModel : 토큰 임베딩 + Transformer blocks + LM head

MultiHeadAttention, FeedForward, TransformerBlock

TextDataset : Hugging Face tokenized dataset → fixed-length blocks

Trainer : 학습/검증 루프, W&B 로깅

generate : 샘플 텍스트 생성 함수

---

## Training Results

### 01/08

<img width="600" height="430" alt="스크린샷 2026-01-06 190207" src="https://github.com/user-attachments/assets/d7315d2f-9bbe-4c84-aadd-a58f0cc3194d" />

val/perplexity	1.9965586613604558

val/loss	0.6914250291883945

### 01/09

<img width="600" height="430" alt="스크린샷 2026-01-07 153144" src="https://github.com/user-attachments/assets/cc1773ff-3a5a-444a-b4ca-d2aea459077d" />

val/perplexity	1.9718610883952392

val/loss	0.6789778117090464

### 01/10

<img width="600" height="430" alt="스크린샷 2026-01-11 172446" src="https://github.com/user-attachments/assets/045e2965-2399-43cb-9863-d66eebebadbc" />

val/perplexity	1.9021691870350403

val/loss	0.642142316326499

### 01/11

<img width="600" height="430" alt="스크린샷 2026-01-11 173228" src="https://github.com/user-attachments/assets/cb278a26-de34-4578-81b1-d8f71eb719a6" />

val/perplexity	1.8892623992905415

val/loss	0.6361864879727364

---

## Issues and Update

### 01/08 (Prototype)

### 01/09 (ReLU -> GELU)

Update : 활성화 함수 ReLU에서 GELU로 변경

Issue : EOS(End Of State)가 반복되는 현상

Update : EOS token id(50256)을 지정해서 EOS가 나오면 break문 실행  

Update : 추가로 Repetition Penalty를 지정하여 최근 50토큰동안 반복된만큼 logit에 반영

### 01/10 (learning rate)

Update : lr 5e-4 -> 7e-4

### 01/11

Update : RoPE 추가 

### 01

Update : Flash Attention 추가 

---

## ROPE란?

---
## Flash Attention이란? 

1. 주요 특징 고속 연산: 기존 어텐션 대비 속도가 최대 수 배 빠릅니다.메모리 절약: 시퀀스 길이(\(N\))에 따른 메모리 복잡도를 \(O(N^{2})\)에서 \(O(N)\)으로 획기적으로 줄여, 더 긴 문맥(Long Context) 처리가 가능합니다.정확도 유지: 근사치(Approximation)를 구하는 방식이 아닌, 수학적으로 동일한 결과를 내는 Exact Attention 방식입니다. 2. 핵심 원리 Tiling (타일링): 대용량의 어텐션 행렬을 한꺼번에 계산하지 않고, 작은 블록(타일) 단위로 나누어 GPU의 빠른 메모리인 SRAM에서 처리합니다.Recomputation (재계산): 중간 결과인 \(N\times N\) 어텐션 행렬을 HBM(메인 메모리)에 저장하는 대신, 역전파(Backpropagation) 시에 필요한 부분만 다시 계산하여 메모리 읽기/쓰기(I/O) 오버헤드를 극대화로 줄입니다.

2. 현재 대부분의 딥러닝 라이브러리에서 기본적으로 지원합니다.
PyTorch: torch.nn.functional.scaled_dot_product_attention을 사용하면 조건 충족 시 자동으로 실행됩니다.
직접 설치: FlashAttention 공식 GitHub에서 소스 코드를 확인하고 설치할 수 있습니다. 
이 기술은 GPT-4, Llama 3와 같은 최신 대형 언어 모델(LLM)이 수만 단어 이상의 긴 문장을 빠르게 처리할 수 있게 만든 핵심 기술 중 하나입니다.
