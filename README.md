# TruthLens AI

A hallucination-aware Retrieval-Augmented Generation (RAG) system over 141 arXiv AI/ML research papers. TruthLens answers questions from the papers, checks every claim in its answer against the source it cites, reports a **Trust Score**, and repairs or withholds answers whose claims fail verification.

**Research question:** can domain-specific LoRA fine-tuning, evidence-based claim verification, trust scoring and self-correction measurably reduce hallucination compared with a standard RAG baseline?

## How it works

```
question
  -> retrieve 5 chunks (bge-base-en-v1.5 embeddings, ChromaDB, 9,828 chunks from 141 papers)
  -> generate a cited answer (Qwen2.5-7B-Instruct, 4-bit, + QLoRA adapter)        Stage 3
  -> split the answer into atomic claims; judge each claim against its cited source  Stage 4
  -> Trust Score from claim support, citation precision and retrieval relevance      Stage 5
  -> if a claim is flagged: regenerate with feedback / prune flagged sentences,
     keep the best verified candidate or decline to answer                           Stage 5
```

Everything runs locally on an 8 GB laptop GPU (RTX 3080); the LoRA adapter was trained on a free Colab T4.

## Ablation stages and results

Every stage is evaluated on the same items: a fixed set of 12 questions (2 should be refused) and a held-out set of 38 items (30 answerable, 8 unanswerable) drawn from 14 papers never seen in training.

| Stage | System | Key result |
|---|---|---|
| 1 | No-RAG (raw Qwen) | Fluent but unsourced; answers out-of-corpus questions from memory and invents facts (e.g. calls GPT-4 an Anthropic model) |
| 2 | Base RAG | Grounded answers, but 25-40% of sentences uncited; only 10/30 held-out answers cite exclusively correct sources |
| 3 | + LoRA fine-tuning | Uncited sentences 25-40% → **0%**; answers with all-correct citations 10/30 → **21/30**; answers much shorter |
| 4 | + claim verification | Correctly cited claims **59% → 85%**; but confirmed unsupported claims 1.3% (base) vs 6.1% (LoRA): fine-tuning improved attribution, **not** grounding |
| 5 | + trust score + self-correction | Trust Score separates clean LoRA answers (mean 91) from ones with a confirmed unsupported claim (mean 25.5); self-correction removed all 6 confirmed unsupported claims, at the cost of declining 3 answerable questions |

### Main findings

1. **Fine-tuning taught citation discipline, not truthfulness.** The LoRA model cites precisely, but when it is wrong its errors look exactly like correct answers (fully cited answers to a false-premise question; misread numbers). Citation format alone is not evidence of support.
2. **Automated verification must itself be validated.** Two plausible verifier designs (NLI alone; Qwen judge + NLI veto) would each have produced a misleading comparison. The adopted Qwen judge approves 94% of truly supported claims (64/68) but catches only 4/10 unsupported ones, so automated hallucination rates are lower bounds. Verifier accuracy was measured on 78 hand-labelled claims, including a 40-claim blind held-out set (36/40 correct).
3. **The old trust score rewarded the wrong thing.** The interim heuristic (retrieval similarity + share of sources cited) scored the base model's *bad* answers higher than its good ones (77 vs 61). The claim-based score fixes this.
4. **Self-correction can withhold a fabrication but cannot supply a missing fact.** When told which claims failed, the model either declined or repeated the error; pruning and refusing did the useful work.

Small samples throughout (50 items; 9 confirmed unsupported claims; single annotator). Differences of 1-2 items are not meaningful and no significance testing is claimed.

## Trust Score

Reported as **"Trust Score: N/100 (High / Moderate / Low)"** with supporting counts, never as a percentage confidence: it is a heuristic and its calibration has not been tested.

| Component | Weight |
|---|---|
| Claim support (supported = 1, true but mis/uncited = 0.75, unsupported = 0) | 60% |
| Citation precision (cited claims whose source supports them) | 15% |
| Retrieval relevance (best source similarity, rescaled 0.50 → 0.80) | 25% |

Any unsupported claim caps the score at 69 (never High); a contradicted claim caps it at 39. Refusals are scored on a separate path and shown as **"Refusal Score: N/100"** (Justified ≥ 70 / Uncertain ≥ 40 / Questionable refusal): it rates how justified declining was, high when retrieval found nothing relevant and low when relevant sources were ignored. It is not a confidence that any answer is correct.

## Quick start (Windows / PowerShell)

```powershell
cd TruthLensAI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python scripts\ingest_all.py          # download papers, extract, clean, chunk
python scripts\embed_all.py           # embed chunks
python scripts\build_vectorstore.py   # build the ChromaDB collection

streamlit run frontend\streamlit_app.py   # demo UI (Ask + Results pages)
python scripts\chat_cli.py                # same pipeline in the terminal
```

The LoRA adapter weights are not in the repository (only their config and training curve). Without them the pipeline falls back to base Qwen as the generator.

### Hosted demo (replay mode)

The live pipeline needs a CUDA GPU with about 6 GB of VRAM, so the hosted demo on Streamlit Community Cloud runs in **replay mode**. When the app finds no CUDA GPU (or `TRUTHLENS_REPLAY=1` is set), the Ask page shows full-pipeline runs recorded locally: 7 demo questions × 4 sidebar settings (LoRA on/off × self-correction on/off), stored in `data/demo/recorded_runs.json`. The Results page is the same as the local app. In replay mode the app imports only `streamlit` (`frontend/requirements.txt`).

```powershell
python scripts\record_demo.py             # re-record on the local GPU (resumable; --fresh to start over)
$env:TRUTHLENS_REPLAY = "1"; streamlit run frontend\streamlit_app.py   # preview replay mode locally
```

To deploy: on [share.streamlit.io](https://share.streamlit.io), create an app from this repository (branch `main`) with main file path `frontend/streamlit_app.py`, and set Python 3.12 under Advanced settings.

### Reproducing the evaluation

```powershell
python scripts\run_baseline_eval.py         # Stage 1 vs 2
python scripts\run_lora_eval.py             # Stage 2 vs 3
python scripts\verify_claims.py             # Stage 4: extract + verify claims (~25 min)
python scripts\evaluate_verifier.py         # verifier accuracy on the hand-labelled gold sets
python scripts\evaluate_trust_score.py      # Stage 5: trust score vs manual labels (no LLM needed)
python scripts\evaluate_self_correction.py  # Stage 5: self-correction on flagged answers
pytest tests -v
```

## Repository layout

```
app/
  ingestion/        PDF download, extraction, cleaning, chunking with page/section metadata
  embeddings/       bge-base-en-v1.5 embedding
  vectorstore/      ChromaDB wrapper
  retriever/        top-k retrieval and citation-tagged context
  rag/              Qwen loading + LoRA, prompts, full TruthLens pipeline (truthlens.py)
  finetuning/       LoRA dataset generation and RAG formatting
  verification/     claim extraction and evidence-based verification
  trust_score/      interim and claim-based trust scores
  self_correction/  regenerate / prune / refuse
  evaluation/       stage comparison harnesses
frontend/           Streamlit demo (live on a GPU, replay mode without one)
data/demo/          recorded pipeline runs for replay mode
notebooks/          Colab QLoRA training notebook
scripts/            CLI entry points
data/eval/          questions, answers, claims, verdicts, gold labels, scores
docs/               evaluation reports and findings
```

## Write-ups

| Document | Contents |
|---|---|
| `docs/evaluation_baseline.md` | Stage 1 vs 2 answers side by side |
| `docs/trust_score_interim.md` | The first heuristic trust score and its flaws |
| `docs/phase8_lora_findings.md` | QLoRA training and Stage 2 vs 3 comparison |
| `docs/phase9_verification_findings.md` | Verifier design, validation and claim-level audit |
| `docs/phase10_11_findings.md` | Claim-based trust score and self-correction |

## Limitations and future work

- The verifier is the same model family as the generator and is lenient (catches 4/10 unsupported claims); a stronger or independent judge is the most valuable upgrade.
- Claim extraction does not resolve pronouns in short sentences, and truncated list items can become claims.
- Self-correction's selection threshold was chosen on the evaluation items; it needs a held-out test. Re-retrieval using the flagged claims is untried.
- The LoRA answers are short (about 23 words) because the training answers were; a retrain with 1 epoch, longer targets and false-premise refusal examples is planned.
- Not yet done: RAGAS / DeepEval evaluation across all stages, a FastAPI backend, Docker deployment, trust score calibration.

## Stack

Qwen2.5-7B-Instruct (4-bit NF4, bitsandbytes) · PEFT / QLoRA · BAAI/bge-base-en-v1.5 · ChromaDB · cross-encoder/nli-deberta-v3-base · Streamlit · pypdf · arXiv API
