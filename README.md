# Decoder-Only Transformer (MiniGPT)

A **decoder-only transformer language model** built from scratch in PyTorch. This project implements a GPT-style architecture with multi-head self-attention, feed-forward layers, and sinusoidal positional encodings for autoregressive text generation.

---

## Project Overview

This model is a **from-scratch implementation of a GPT-like transformer**. It can:

- Process tokenized sequences and learn autoregressive relationships.
- Generate text by predicting the next token in a sequence.
- Handle variable-length sequences with padding.
- Train on custom datasets using PyTorch `DataLoader`.

---

## Model Architecture

### 1. Multi-Head Self-Attention
- Splits input embeddings into multiple heads.
- Computes scaled dot-product attention for each head.
- Concatenates results and applies linear projection.
- Dropout included for regularization.

**Input:** `(B, T, d_model)`  
**Output:** `(B, T, d_model)`  

Where:  
- `B` = batch size  
- `T` = sequence length  
- `d_model` = model embedding size  

---

### 2. Feed-Forward Network (FFN)
- Two linear layers with GELU activation.
- Dropout for regularization.

---

### 3. Decoder Block
- LayerNorm → Multi-Head Attention → Residual  
- LayerNorm → Feed-Forward → Residual  
- Multiple blocks stacked to form the full transformer decoder.

---

### 4. Positional Encoding
- Sinusoidal positional encoding added to embeddings.
- Captures sequential information without recurrence.

---

### 5. MiniGPT
- Token embedding + positional encoding
- `n_layers` of decoder blocks
- LayerNorm + Linear output projection
- Weight tying between input embeddings and output layer
- Causal masking for autoregressive attention
- Cross-entropy loss for next-token prediction

---

## Data Preprocessing

- Tokenizes text with a regex-based tokenizer.
- Builds vocabulary including special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<SEP>`, `<UNK>`.
- Converts prompt-story pairs into token IDs.
- Prepares PyTorch `DataLoader` with padding and shifted targets.

---

## Special Tokens

| Token | Purpose |
|-------|---------|
| `<PAD>` | Padding token |
| `<BOS>` | Beginning of sequence |
| `<EOS>` | End of sequence |
| `<SEP>` | Separator between prompt and story |
| `<UNK>` | Unknown token |

---

## Configuration Example

```python
cfg = GPTConfig(
    vocab_size=vocab_size,
    d_model=384,
    n_heads=6,
    n_layers=6,
    d_ff=1536,
    dropout=0.1,
    max_len=256,
    pad_id=vocab[PAD]
)
model = MiniGPT(cfg).to(device)

