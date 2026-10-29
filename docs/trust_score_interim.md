# Interim Trust Score: Observed Behaviour

Recorded: 2026-10-29 via `scripts\chat_cli.py` (Base RAG, Qwen2.5-7B-Instruct 4-bit, top_k=5, threshold 0.3).

Implementation: `app/trust_score/trust_score.py`
- Non-refusal: `round(100 * (0.65 * retrieval_confidence + 0.35 * citation_coverage))`
- Refusal: `min(round(15 + 10 * (1 - retrieval_confidence)), 30)`
- Labels: High >= 70, Moderate >= 40, Low < 40

The score is a heuristic, not a probability. Calibration is untested until Phase 12.

---

## Results

| Query | In corpus? | Refusal | Retrieval conf. | Citation cov. | Trust Score |
|---|---|---|---|---|---|
| What is football? | No | Yes | 0.442 | 0.0 (0/5) | **21/100 (Low)** |
| What causes hallucinations in large language models? | Yes | No | 0.815 | 0.8 (4/5) | **81/100 (High)** |

Both match the intended behaviour: low for out-of-corpus, high for in-corpus.

VRAM during the session: 5.63 GB allocated after model load, 6.14 GB reserved peak.

---

## Observations

**1. The retrieval threshold did not filter the out-of-corpus query.**
All five "football" chunks scored 0.437-0.450 similarity, above the 0.3 threshold. bge-base similarities are compressed into a high range, so unrelated text still clears 0.3. The refusal came entirely from the grounding prompt (the second line of defence). A threshold around 0.5-0.55 would have caught this query, but it needs tuning against in-corpus queries before changing.

**2. A high score did not mean a fully faithful answer.**
The 81/100 answer contains:
- An uncited claim: "Intrinsic hallucinations arise from the model itself, while extrinsic hallucinations might be influenced by external factors." This is not attributed to any source and does not match the standard definitions (intrinsic = contradicts the source; extrinsic = cannot be verified from the source).
- An admission that "the sources do not provide specific causes", so the question was only partially answered.

**3. Refusal detection has false positives.**
`REFUSAL_RE` matches phrases anywhere in the answer, so a correct answer that *describes* a limitation is scored as a refusal and capped at 30. Found during Phase 8.2: "...LLMs cannot answer questions about events that occurred after they were trained..." matches `cannot answer`. Phase 10 should detect refusals at the answer level (for example, only in the opening sentence, or only when the answer has no citations) rather than by substring.

The interim score cannot detect issues 2 or 3 because it measures retrieval quality and citation *presence*, not whether each claim is supported. This is the motivating case for claim-level verification (Phase 9) and the claim-support ratio in the full trust score (Phase 10).
