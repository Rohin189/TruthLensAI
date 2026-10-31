# Phase 10-11 Findings: Claim-Based Trust Score and Self-Correction

Recorded: 2026-10-31. Inputs: the Phase 8 answers and Phase 9 claim verdicts (`data/eval/verification_phase8.jsonl`); 12 fixed questions + 38 held-out items, Base RAG and LoRA RAG.
Outputs: `data/eval/trust_scores_phase8.jsonl`, `docs/evaluation_trust_score.md` (auto-generated), `data/eval/self_correction_phase8.jsonl`.

## Phase 10: claim-based trust score

### Why the interim score had to go

The interim score (`compute_trust_score`) blended mean retrieval similarity (65%) with *citation coverage*, the share of the 5 retrieved sources that the answer cites (35%). It measured whether an answer cites many sources, not whether those sources support it: the LoRA model's score fell 69.6 → 57.0 while its citation accuracy doubled, and an 81/100 answer contained an uncited, incorrect definition (`docs/trust_score_interim.md`).

### Design (`app/trust_score/trust_score.py`, `compute_claim_trust_score`)

| Component | Weight | Definition |
|---|---|---|
| Support | 0.60 | Mean credit per claim: supported 1.0; misattributed or uncited-but-supported 0.75 (true content, wrong attribution); unsupported or contradicted 0 |
| Citation precision | 0.15 | Supported claims / claims that cite a source |
| Retrieval relevance | 0.25 | Best source similarity rescaled linearly from 0.50 → 0 to 0.80 → 1 |

- **Caps:** any unsupported claim caps the score at 69 (an unverifiable claim rules out "High"); any contradicted claim caps it at 39 ("Low").
- **Refusals** (refusal phrase *and* no citations, `is_refusal_answer`) score how justified declining is: `100 × (1 − relevance)`. Declining when the sources are off-topic scores high; declining when strong sources were retrieved scores low.
- **Relevance scaling** comes from measured similarities on the 50 eval items: off-topic questions ("What is football?") top out at 0.45-0.52 because bge similarities are compressed, while answerable questions reach 0.67-0.87. The interim score used the raw mean, so even "What is football?" received 0.44 × 65 ≈ 29 points from retrieval.
- Reported as **"Trust Score: N/100 (Label) - k/n claims supported by their cited source"**. The weights are design choices, not fitted; calibration is untested (Phase 12).

### Evaluation: does the score separate good answers from bad ones?

Reference labels: the Phase 9 manual adjudication, which found a confirmed unsupported claim in 3 base and 4 LoRA answers. Full table: `docs/evaluation_trust_score.md`.

| Group | n | Interim | Claim-based |
|---|---|---|---|
| Base: answered, no confirmed unsupported claim | 38 | 61.4 | 84.7 |
| Base: answered, confirmed unsupported claim | 3 | **77.0** | 69.0 |
| LoRA: answered, no confirmed unsupported claim | 38 | 55.9 | **91.2** |
| LoRA: answered, confirmed unsupported claim | 4 | 54.2 | **25.5** |
| Correct refusals (base / LoRA) | 9 / 8 | 18.6 / 18.6 | 46.7 / 47.5 |

1. **The interim score ranked the base model's bad answers above its good ones** (77.0 vs 61.4): long answers citing many sources score well whether or not the sources support them. The claim-based score reverses this.
2. **For the LoRA model the separation is large** (91.2 vs 25.5): all 4 answers with a confirmed unsupported claim score Low (19-34). Its answers are short (2.4 claims), so one bad claim dominates.
3. **The base model's bad answers only drop to 69 (Moderate)**, the unsupported-claim cap: one bad claim among 6-11 good ones barely moves the mean, so the cap does the work. Four clean base answers were also capped because of checker false alarms; the checker's precision on flagged claims is low (9 of 24 confirmed in Phase 9), which is why a flag caps at Moderate rather than forcing Low.
4. **Refusal scores are moderate (~47)** because the held-out refusal items retrieve on-topic chunks from other papers (similarity 0.60-0.78). The score cannot tell "sources are about the topic but do not answer this question" from "sources answer it": a limitation of a relevance-only refusal score.
5. The score inherits the checker's blind spot: it caught only 4/10 unsupported gold claims, so a High score means "no claim was flagged", not "no claim is wrong".

## Phase 11: self-correction

### Design (`app/self_correction/corrector.py`)

Triggered only when at least one claim is flagged unsupported or contradicted. Two repairs are built:

- **regenerate:** same sources; the prompt lists the flagged statements and asks for an answer that drops or corrects them. The new answer is re-verified from scratch.
- **prune:** delete the sentences that produced flagged claims; the surviving claims keep their verdicts. If nothing survives, the answer becomes a refusal.

**Selection:** an answer candidate is eligible only if at least half its claim credit is verified (`MIN_SUPPORT_TO_ANSWER = 0.5`) and no claim is contradicted; the highest-scoring eligible candidate wins (ties → least edited). If none is eligible, a refusal is returned.

The first version simply took the highest trust score. Reviewing its output showed the flaw: a refusal scores `100 × (1 − relevance)`, which is low when relevant sources were retrieved, so a confirmed-wrong answer (q10: 24/100) beat an honest refusal (5/100). The eligibility rule fixes this. The no-contradiction condition was added after a live false-premise question ("Why does LoRA *increase* the number of trainable parameters...?") kept an answer whose cited source said the opposite (2/3 claims supported, 39/100); it changes none of the 7 results below. **It was chosen after seeing these 7 items and has not been validated on held-out data.**

Re-retrieval (querying again with the flagged claims) was not implemented: retrieval for the same question is deterministic, and the flagged facts were usually absent from the sources rather than mis-retrieved.

### Results on the 7 flagged LoRA answers

The other 35 LoRA answers had no flagged claim and pass through unchanged. Each candidate below was compared by hand with the Phase 9 adjudication (and, for the new val08 answer, with its source).

| Item | Original answer (manual label) | Trust Scores: original / regenerate / prune | Kept | Outcome |
|---|---|---|---|---|
| q10 | Confirmed unsupported (challenges not in source) | 24 / 5 (refusal) / 5 (refusal) | refusal | Error removed; answerable question now declined |
| val08 | Correct, checker false alarm | 23 / **98** / 10 (refusal) | regenerate | New answer verified by hand: verbatim from Source 1, more precise (16.7%, per-model rates) |
| val09 | Confirmed unsupported (invented "prompting") | 25 / 0 (refusal) / 0 (refusal) | refusal | Error removed; answerable question now declined |
| val10 | Correct, claim-extraction defect | **69** / 20 (refusal) / 20 (refusal) | original | Correct answer preserved |
| val13 | Confirmed unsupported (false-premise question) | 34 / 40 (refusal) / 40 (refusal) | refusal | **Correct behaviour**: Base RAG also refused this question |
| val22 | Correct, checker false alarm | **65** / 42 (refusal) / 42 (refusal) | original | Correct answer preserved |
| val38 | Confirmed unsupported (misread numbers) | 19 / 19 (same error) / 23 (refusal) | refusal | Error removed; answerable question now declined |

**Summary**

- **All 6 confirmed unsupported claims in Stage 4 output were removed** (4 answers); the 3 correct-but-flagged answers were preserved (2 unchanged, 1 replaced by an equally correct, better-cited answer).
- **The cost is coverage:** 3 answerable questions (q10, val09, val38) are now declined rather than answered wrongly. The false-premise question val13 is now refused, as it should be (LoRA refusal accuracy on the held-out set 6/8 → 7/8).
- **Regeneration on its own fixed none of the 4 real errors.** Told which statements failed, the LoRA model either declined (3) or repeated the same wrong numbers with a different citation (val38). The model does not have the missing fact, so it cannot repair it, only withhold it. Pruning and refusal did the useful work; regeneration helped only where the original was a false alarm (val08).
- In this setting self-correction trades answers for honesty: it turns confident fabrications into refusals. Whether a better-informed repair (re-retrieval with the flagged claim, a larger context) can recover the correct answer is untested.

## Limitations

- **Tiny sample:** 7 corrected answers, 4 with a confirmed error. This is a demonstration of the mechanism, not evidence of a general improvement.
- **The judge grades its own repairs.** Candidates are selected with the same verifier that flagged them, so a corrected answer's trust score is optimistic by construction. Every corrected answer was therefore checked by hand against its sources.
- **Unflagged errors pass through untouched.** Self-correction only sees what the checker flags, and the checker caught 4/10 unsupported gold claims.
- **The selection threshold was tuned on the evaluation items** (see above).
- **Latency:** a flagged answer costs one extra generation plus a second verification pass (roughly 10-30 s on the RTX 3080).
