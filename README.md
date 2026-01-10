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

## Training Results

### 01/08

###

## Issues and Update

### 01/09

Issue : EOS(End Of State)가 반복되는 현상

Update : EOS token id(50256)을 지정해서 EOS가 나오면 break문 실행  

Update : 추가로 Repetition Penalty를 지정하여 최근 50토큰동안 반복된만큼 logit에 반영


### 01/10

Update : RoPE 추가 

Result : 

Update : Flash Attention 추가 
