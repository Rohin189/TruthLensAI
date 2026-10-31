---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
---

# TruthLens QLoRA adapter for Qwen2.5-7B-Instruct

A LoRA adapter that teaches Qwen2.5-7B-Instruct to answer from retrieved research-paper excerpts with a `(Source N)` citation on every sentence, and to say so when the sources do not answer the question. It is the Stage 3 generator of [TruthLens AI](../../../README.md).

**The weights (`adapter_model.safetensors`) are not in the repository**; only this card, `adapter_config.json`, `training_run.json` and `training_curve.png` are committed.

## Training

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct, 4-bit NF4 (QLoRA) |
| LoRA | r = 16, alpha = 32, dropout 0.05, all attention and MLP projections |
| Data | 321 train / 38 validation RAG-format examples from arXiv AI/ML papers, split by paper (`data/finetuning/`) |
| Loss | Answer tokens only (about 3.5% of tokens) |
| Optimiser | lr 2e-4, cosine schedule, effective batch 8, seed 42, fp16 |
| Hardware | Google Colab T4, peak VRAM 11.97 GB |
| Kept checkpoint | checkpoint-40 (end of epoch 1, best validation loss 0.479; later epochs overfit) |
| Versions | transformers 4.45.2, peft 0.13.2 |

Prompts are built with `build_prompt` in `app/rag/pipeline.py` (5 sources, single user turn), the same format used at inference.

## Results (vs the base model on identical retrieved sources)

- Uncited sentences: 25-40% → 0%.
- Held-out answers whose citations are all correct: 10/30 → 21/30.
- Correctly cited claims: 59% → 85%.
- Not improved: grounding. After manual checking, 6/99 claims were unsupported vs 3/234 for the base model, including confident answers to a false-premise question.
- Answers are short (about 23 words) because the training answers were.

Details: `docs/phase8_lora_findings.md`, `docs/phase9_verification_findings.md`.

## Usage

```python
from app.rag.llm import attach_lora_adapter, generate
attach_lora_adapter("models/lora_adapters/qwen25-7b-truthlens-lora")
answer = generate(prompt, temperature=0)
```

Requires the pinned versions above so the adapter loads; inference fits in 8 GB VRAM (about 5.7 GB).
