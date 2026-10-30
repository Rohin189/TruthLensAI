# Phase 8 Findings: LoRA-fine-tuned RAG (Stage 3) vs Base RAG (Stage 2)

Recorded: 2026-10-31. Raw results: `data/eval/results_lora.jsonl`, `data/eval/results_lora_val.jsonl`; per-question answers: `docs/evaluation_lora.md`.

## Setup

**Training** (`notebooks/phase8_qlora_train.ipynb`, Colab T4):
- Qwen2.5-7B-Instruct, 4-bit NF4, LoRA r=16 / alpha=32 on all attention and MLP projections.
- Data: 321 train / 38 val RAG-format examples (Phase 8.2), split by paper. Loss on the answer only (about 3.5% of tokens).
- 3 epochs, 120 steps, 89.5 min, peak VRAM 11.97 GB. transformers 4.45.2, peft 0.13.2.

| Epoch | Train loss | Val loss |
|---|---|---|
| 0.5 | 0.627 | 0.592 |
| **1.0** | **0.512** | **0.479 (best, kept)** |
| 1.5 | 0.343 | 0.487 |
| 2.0 | 0.297 | 0.518 |
| 2.5 | 0.120 | 0.584 |
| 3.0 | 0.178 | 0.601 |

Validation loss bottoms out after 1 epoch and then rises, so the model overfits from epoch 2 onwards. The saved adapter is checkpoint-40 (end of epoch 1). With about 13k answer tokens of data, **1 epoch is sufficient**.

**Evaluation** (`scripts/run_lora_eval.py`): both systems get identical retrieved sources and decode greedily from the same 4-bit weights; only the adapter differs. Base RAG was re-run under these conditions rather than reusing the temperature-0.3 Phase 7 results.

## Results

**Fixed question set (12):**

| Metric | Base RAG | LoRA RAG |
|---|---|---|
| Correct answer/refuse behaviour | 12/12 | 12/12 |
| Uncited sentence rate (answerable) | 0.252 | **0.000** |
| Invalid citations | 0 | 0 |
| Interim Trust Score, answerable (mean) | 69.6 | 57.0 (see finding 3) |
| Interim Trust Score, refusals (mean) | 19.5 | 19.5 |
| Mean words | 107.0 | 22.8 |

**Held-out validation set (38; 30 answerable, 8 refusal; papers unseen in training):**

| Metric | Base RAG | LoRA RAG |
|---|---|---|
| Answered when answerable | 30/30 | 30/30 |
| Cites the gold chunk | 23/30 | 21/30 |
| Cites a chunk from the gold paper | 29/30 | 30/30 |
| **All citations are gold** | 10/30 | **21/30** |
| Content recall vs reference answer | 0.717 | 0.647 |
| Uncited sentence rate | 0.401 | **0.000** |
| Refused when unanswerable | 7/8 | 6/8 |
| Mean words | 78.0 | 25.7 |

## Findings

**1. Citation discipline improved sharply.** Every LoRA sentence carries a citation (uncited rate 25% to 0% on the fixed set, 40% to 0% on validation), and the share of answers whose citations are all correct doubled (10/30 to 21/30). Base RAG often states facts without attribution or cites sources that do not contain the claim.

**2. q7 is now answered correctly.** Base RAG blends two conflicting definitions (one from Source 2, one from Source 1), two of its three sentences are uncited, and earlier sampled runs gave an incorrect definition (`docs/trust_score_interim.md`). LoRA RAG gives the standard definition in one cited sentence: intrinsic = contradicts the input, extrinsic = cannot be verified from the source.

**3. The interim Trust Score drop is a metric artefact, not a quality drop.** Citation coverage is defined as the fraction of the 5 retrieved sources that are cited, so a precise answer that cites only the one supporting source is penalised. Example q1: retrieval confidence is identical for both systems (0.815); Base cites 4/5 sources and scores 81, LoRA cites 1/5 and scores 60 (`0.65 x 0.815 + 0.35 x 0.2`). The interim score rewards citing many sources rather than citing correctly. **Phase 10 must replace coverage-of-retrieved-sources with a claim-support ratio.**

**4. Trade-off: brevity and completeness.** LoRA answers collapsed to a single sentence (about 23 words) for every question, including broad ones such as q9 (RAG evaluation metrics), where Base RAG also covered LLM judges and online telemetry. Content recall fell from 0.717 to 0.647. Cause: the Phase 8.1 training answers are short (mean 215 characters, mostly 1-2 sentences), so the model learned that length. LoRA did surface one grounded detail Base missed (MRR, from a metrics table in Source 5).

**5. One new failure: false-premise acceptance.** On a validation refusal ("How does the proposed framework achieve its goal without requiring gradient updates?"), Base RAG correctly refused, but LoRA answered using a loosely related source (Federated Sketching LoRA). "Sketch" and "client" appear in that source; "without gradient updates", "communication overhead" and "server" do not. The model accepted the question's premise and produced a partly unsupported answer with a confident citation. A second refusal case was failed by both systems; its question ("the primary issue the proposed framework aims to address in vision-language models") is vague enough that the label is arguable.

## Conclusion

Fine-tuning changes **how** the model answers more than **what** it knows: attribution becomes consistent and more precise, at the cost of completeness on broad questions and one instance of confidently cited, premise-accepting hallucination. Citations alone therefore do not guarantee support. This is direct motivation for Phase 9 claim-level verification, which checks each claim against its cited source.

## Limitations

- Small samples (12 + 38 items, only 8 refusal cases); differences of 1-2 items are not meaningful and no significance testing was done.
- Rule-based proxy metrics; semantic evaluation (RAGAS/DeepEval) is deferred to Phase 12.
- `is_refusal` uses the interim `REFUSAL_RE`, which has known false positives.
- Training and validation questions were generated by the same base model whose behaviour is being compared.

## Recommendations for a retrain (optional)

- `EPOCHS = 1` (same best checkpoint, about 60 minutes less Colab time).
- Add multi-sentence targets for broad questions (several gold chunks, one claim per sentence) to recover completeness.
- Add false-premise refusal examples (a question asserting something the sources do not support).
