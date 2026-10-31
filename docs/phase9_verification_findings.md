# Phase 9 Findings: Claim-Level Verification of Stage 2 (Base RAG) and Stage 3 (LoRA RAG)

Recorded: 2026-10-31. Inputs: the Phase 8 answers (`data/eval/results_lora.jsonl`, `results_lora_val.jsonl`): 12 fixed questions + 38 held-out validation items, each answered by Base RAG and LoRA RAG from identical retrieved sources.

Outputs: `data/eval/claims_phase8.jsonl` (claims), `data/eval/verification_phase8.jsonl` (verdicts), `data/eval/verifier_gold.jsonl` and `verifier_heldout.jsonl` (hand-labelled checker test sets), `data/eval/verifier_predictions.jsonl`.

## 1. Method

**Claim extraction** (`app/verification/claim_extractor.py`). Answers are split into sentences (list items, paragraphs and abbreviations handled). Each claim inherits the `(Source N)` citations of its sentence; citation text is stripped from the claim, including mid-sentence attributions such as "as defined in Source 1". Sentences of 25+ words are split into atomic claims by Qwen; a split is rejected if any claim is under 3 words, mentions a source number, or has under 70% of its content words drawn from the sentence or question. An answer is a refusal only if it contains a refusal phrase **and** no citations (fixing the interim trust score's substring false positive).

| | Base RAG | LoRA RAG |
|---|---|---|
| Answers (non-refusal) | 41 | 42 |
| Refusals | 9 | 8 |
| Claims | 234 (5.7 / answer) | 99 (2.4 / answer) |
| Uncited claims | 79 | 0 |

**Verification** (`app/verification/verifier.py`). Each claim is judged against the source(s) it cites. Verdicts: `supported` (cited source supports it), `misattributed` (another retrieved source does), `contradicted`, `unsupported`, `uncited_supported` (no citation, but a retrieved source supports it).

- **Primary judge:** Qwen2.5-7B-Instruct with base weights (adapter disabled), seeing only the question (for resolving "it"/"this"), the claim and the cited passage(s), and replying with a label and a one-line reason. Greedy decoding.
- **Sensitivity check:** the same verdicts with an NLI veto (`cross-encoder/nli-deberta-v3-base`): a Qwen SUPPORTED label is overridden when NLI entailment < 0.01. Both verdicts are stored per claim.

## 2. How the verifier was chosen (and why the design changed twice)

1. **NLI alone (the original, user-approved design) failed.** On the 38-claim dev set it approved only 15/30 truly supported claims at the standard 0.5 threshold, including near-verbatim ones (entailment 0.00-0.03). Causes: sentence windows lose context ("the process" = LoRA-SMoE), and the model is strict about paraphrase and hedging in dense scientific text. Its best agreement required a threshold of 0.01 tuned on the same items, a sign of poor calibration in this domain.
2. **Qwen + NLI veto looked best on the dev set** (32/38; caught 7/8 unsupported) and was adopted.
3. **A manual audit of vetoes on real answers found 10 of 13 wrong** (8/10 sampled base vetoes, 2/3 LoRA vetoes). The veto fired on 27/234 base claims but only 3/99 LoRA claims, because base claims are longer and more paraphrased. It therefore inflated the base model's unsupported rate (16.2% with veto vs 6.4% without) and biased the Stage 2 vs Stage 3 comparison. The veto was demoted to a sensitivity check.

## 3. Verifier accuracy

Gold labels judge each claim against its cited source only. Rule: supported if the cited text states the claim or clearly implies it in one step; not supported if the claim adds facts, inverts a relation, overstates, or uses wording from a different chunk. Eleven borderline labels were re-reviewed in a second pass; three changed (G22, H13, H23, all to supported).

| Policy | Dev (38; oversampled failure cases) | Held-out (40; random, labelled blind) | Supported approved (both) | Unsupported caught (both) |
|---|---|---|---|---|
| **Qwen judge (primary)** | 32/38 | **36/40** | **64/68 (94%)** | **4/10** |
| Qwen + NLI veto | 31/38 | 36/40 | 60/68 (88%) | 7/10 |

**Interpretation:** the judge reliably confirms supported claims but is lenient on subtle overstatement and on wording borrowed from a neighbouring chunk. Only 10 negative examples exist across both sets (3 in held-out), so the 40% catch rate is highly uncertain. **Automated unsupported rates below are therefore lower bounds on true unsupported content.** Known judge failure modes observed: misreading a correct claim as contradicted (val08), rejecting a claim the source states directly (val22), and inventing a justification (dev G17: a "Shallow" table category that does not exist).

## 4. Results

### 4.1 Automated verdicts

| Metric (non-refusal answers) | Base RAG | LoRA RAG |
|---|---|---|
| **Correctly cited** (supported / claims) | 59.0% | **84.8%** |
| Grounded (supported by any retrieved source) | 93.6% | 90.9% |
| Flagged unsupported + contradicted | 6.4% (15) | 9.1% (9) |
| Misattributed claims | 7 | 6 |
| Uncited but supported claims | 74 | 0 |
| Mean per-answer claim-support ratio | 0.597 | 0.786 |
| Answers with ≥ 1 flagged claim | 9/41 | 7/42 |

Sensitivity (with NLI veto): correctly cited 50.4% vs 81.8%; flagged 16.2% (38) vs 12.1% (12). By subset (primary): fixed questions, flagged 6.4% vs 5.9%; validation, 6.4% vs 9.8%.

### 4.2 Manual adjudication of every flagged claim

Because the checker's precision on flagged claims proved uneven, all 24 primary-flagged claims were read against their sources.

| | Base RAG | LoRA RAG |
|---|---|---|
| Flagged | 15 | 9 |
| Confirmed unsupported | **3** (q1, q5, q9) | **6** (q10, val09, val38, val13 ×3) |
| Checker errors | 11 | 2 (val08, val22) |
| Claim-extraction defects (truncated list) | 1 (val10) | 1 (val10) |
| **Confirmed unsupported / all claims** | **1.3%** (3/234) | **6.1%** (6/99) |
| Answers with a confirmed unsupported claim | 3/41 | 4/42 |

Most base-model checker errors involve its longer, multi-source, often uncited claims (for example B12-B14, which S2 states directly), which the judge handles worse than LoRA's short single-source sentences.

The LoRA model's confirmed errors are qualitatively serious: three claims in val13 answer a false-premise question ("...without requiring gradient updates?") that Base RAG correctly refused; val38 misreads numbers (gains "from 2.3% to 21.7%" where the source reports 210-233%); val09 adds a method ("prompting") the source does not list.

## 5. Conclusions

1. **Fine-tuning improved attribution, robustly.** Correctly cited claims rose from 59% to 85% (50% to 82% with the veto); uncited claims fell from 74 to 0. This confirms the Phase 8 finding with a claim-level measurement.
2. **Fine-tuning did not reduce hallucination.** Grounding is similar (94% vs 91% of claims supported by some retrieved source). After adjudication the LoRA model produced more confirmed unsupported claims (6 vs 3) from fewer total claims, including confident, fully cited answers to a false-premise question. The counts are small and come from a single annotator, so this is weak evidence of an increase, but there is no evidence of a decrease.
3. **The base model's typical failure is sloppy attribution; the LoRA model's is confident fabrication.** Base RAG mostly says true things but cites them loosely or not at all. LoRA RAG cites precisely and is usually right, but when wrong, its error looks exactly like a correct answer. This is the strongest argument in the project for claim-level verification (Stage 4) and self-correction (Phase 11): citation format alone is not evidence of support.
4. **Automated verification needs auditing.** Two plausible designs (NLI alone; Qwen + veto) would each have produced a misleading Stage 2 vs Stage 3 comparison. The adopted judge is useful for confirming support but undercounts subtle unsupported content and misjudges some long, multi-source claims.

## 6. Limitations

- Small samples: 50 items; 9 confirmed unsupported claims in total; 10 negative gold examples. No significance testing.
- Single annotator (the gold labels and the adjudication were done in this session); the rule used is documented above, but inter-annotator agreement was not measured.
- The judge is the same model family as both generators (adapter disabled when judging), so shared blind spots are possible; its leniency (4/10 unsupported caught) is the measured symptom.
- Claim extraction does not resolve pronouns in sentences under 25 words (they are never split), so claims like "It requires high computational resources" reach the judge without their referent (H13 / B04).
- Each claim is judged only against chunks retrieved for the original question; facts present elsewhere in a paper count as unsupported.

## 7. Implications for later phases

- **Phase 10 trust score:** use the claim-support ratio from the primary judge; weight "unsupported" and "contradicted" verdicts with the judge's measured precision in mind; never treat a high citation count as support. The refusal fix in `is_refusal_answer` should replace `REFUSAL_RE` in the trust score.
- **Claim extraction:** pass the preceding sentence to the judge (or decontextualize short sentences) to fix unresolved pronouns; drop truncated list fragments.
- **Phase 11 self-correction:** target the confirmed LoRA failure modes, namely false-premise questions and number misreading.
- **Optional retrain:** add false-premise refusal examples; the verifier can now measure whether that helps, given the caveats above.
