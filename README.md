# LoRA from Scratch — Parameter-Efficient Fine-Tuning

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-ffd21e?logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/No_peft-From_Scratch-blue" alt="From Scratch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

A **complete from-scratch implementation** of [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) in pure PyTorch — no `peft` library. Build the adapter module, inject it into GPT-2, train it on a real legal NLP task, and match full fine-tuning's accuracy with **0.18% of the parameters**.

```
frozen W (82M params, never trained)  +  B·A (147K params, the only thing you train)
         ↓                                           ↓
   same clause tagger                     613 KB adapter file (vs 327 MB full fine-tune)
```

## Key Results

| Method | Accuracy | Trainable Params | Shipped Per Task |
|--------|----------|-----------------|------------------|
| Full fine-tuning | 95.8% | 81,912,576 (100%) | 327 MB |
| **LoRA r=8** | **95.8%** | **147,456 (0.18%)** | **613 KB** |
| QLoRA (absmax) | ~91.7% | 147,456 | 613 KB + 4-bit base |
| QLoRA (NF4) ★ | ~93.3% | 147,456 | 613 KB + 4-bit base |

★ *NF4 is an innovation in this project — implemented from scratch using the QLoRA paper's normal-float quantization scheme.*

## Innovations Beyond the Standard Implementation

This isn't a copy of a tutorial. Key additions:

1. **NF4 Quantization** — Normal Float 4-bit quantization from the QLoRA paper, implemented from scratch with proper quantile computation, side-by-side with baseline absmax
2. **Multi-Target Injection** — Adapt not just Q/K/V (`c_attn`) but also output projection (`c_proj`) and MLP layers, with preset configurations
3. **Multiple Init Strategies** — Kaiming, Gaussian, and Xavier initialization for the A matrix, with comparison
4. **Dynamic Rank Pruning** — Post-training SVD-based compression: drop low-magnitude rank directions to shrink adapters further without retraining
5. **Spectral Analysis** — SVD decomposition of trained adapters to understand what each layer actually learned and whether it's over-parameterized
6. **Full CLI Pipeline** — Train, evaluate, and analyze from the command line with configurable hyperparameters

## Project Structure

```
lora-from-scratch/
├── src/lora_scratch/           # Core library
│   ├── __init__.py
│   ├── core.py                 # LoRALinear module (the adapter)
│   ├── injection.py            # Freeze model + inject adapters
│   ├── merge.py                # Merge/unmerge/swap adapters
│   ├── quantization.py         # 4-bit quantization (absmax + NF4)
│   ├── data.py                 # LEDGAR dataset loading + processing
│   └── analysis.py             # Parameter counting, spectral analysis, ablation
│
├── lessons/                    # 8 self-contained lessons (run in order)
│   ├── lesson_01_cost_of_full_finetuning.py
│   ├── lesson_02_math_of_low_rank.py
│   ├── lesson_03_build_lora_linear.py
│   ├── lesson_04_inject_into_gpt2.py
│   ├── lesson_05_train_adapter.py      # ~15 min on CPU
│   ├── lesson_06_evaluate.py
│   ├── lesson_07_merge_and_swap.py
│   └── lesson_08_qlora.py
│
├── scripts/                    # CLI tools
│   ├── train.py                # Train with custom hyperparameters
│   ├── evaluate.py             # Evaluate any adapter
│   └── analyze.py              # Spectral analysis + rank ablation
│
├── tests/
│   └── test_lora.py            # Unit tests
│
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repo
git clone https://github.com/SWAYAM109/lora-from-scratch.git
cd lora-from-scratch

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Lessons (Recommended for Learning)

Each lesson is self-contained and builds on the previous one:

```bash
# Lessons 1-4: Concepts + building the adapter (~2 min each, downloads model once)
python lessons/lesson_01_cost_of_full_finetuning.py
python lessons/lesson_02_math_of_low_rank.py
python lessons/lesson_03_build_lora_linear.py
python lessons/lesson_04_inject_into_gpt2.py

# Lesson 5: Train the adapter (~15 min on CPU)
python lessons/lesson_05_train_adapter.py

# Lessons 6-8: Evaluate, merge/swap, quantize (~2-5 min each)
python lessons/lesson_06_evaluate.py
python lessons/lesson_07_merge_and_swap.py
python lessons/lesson_08_qlora.py
```

### 3. Use the CLI Scripts

```bash
# Train with custom rank and targets
python scripts/train.py --rank 16 --alpha 32 --targets attention_full

# Evaluate an adapter
python scripts/evaluate.py --adapter outputs/lora-r8-attention_qkv/lora-adapter.pt

# Analyze adapter internals
python scripts/analyze.py --adapter outputs/lora-training/lora-adapter.pt
```

### 4. Run Tests

```bash
python -m pytest tests/test_lora.py -v
```

## Course Map

| # | Lesson | What You Build | Key Concept |
|---|--------|---------------|-------------|
| 1 | Why not just fine-tune everything? | Memory calculator | 16 bytes/param wall; LoRA's bet |
| 2 | The math of low-rank | SVD energy analysis | ΔW = B·A with r ≪ d; rank is the dial |
| 3 | Build LoRALinear from scratch | The adapter module | Random A, zero B, α/r scaling |
| 4 | Inject it into GPT-2 | Model surgery | Freeze base, wrap c_attn, 0.18% trainable |
| 5 | Train just the adapter | The fine-tune | 147K params → loss 2.59 → 0.057 |
| 6 | Does it match full fine-tuning? | Head-to-head eval | 95.8% vs 95.8%, 613 KB vs 327 MB |
| 7 | Merge & swap adapters | Deployment toolkit | Fold B·A into W; hot-swap tasks |
| 8 | QLoRA | 4-bit compression | Absmax vs NF4; 70B on one GPU |

## How LoRA Works

```
                    ┌─────────────────────┐
                    │  Pretrained Weight W │  ← FROZEN (never updated)
                    │     [768 × 2304]     │
                    └────────┬────────────┘
                             │
        x ──────────────────►├──── W·x ────────────────►  +  ──► output
                             │                              ↑
                             │    ┌───────┐   ┌────────┐   │
                             └───►│ A     │──►│   B    │───┘
                                  │[8,768]│   │[2304,8]│    × (α/r)
                                  └───────┘   └────────┘
                                  ↑ TRAINABLE   ↑ TRAINABLE
                                  (random init) (zero init)
```

**Why it works:**
- Fine-tuning updates live in a low-rank subspace (empirically validated)
- Rank 8 captures ~100% of a realistic weight update
- B = 0 at init → adapter starts as an exact no-op (safe injection)
- α/r scaling decouples adapter strength from rank choice

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- Datasets ≥ 2.14
- NumPy ≥ 1.24
- SciPy ≥ 1.10

**No GPU required.** Everything runs on CPU. The model (distilgpt2, 82M params) and dataset (LEDGAR) download automatically from the Hugging Face Hub — no API keys needed.

## Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

## License

MIT — see [LICENSE](LICENSE).
