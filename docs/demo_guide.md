# TruthLens AI: Demo Guide

About 10 minutes. Every output below was produced by the live pipeline on 2026-11-01 with greedy decoding, so the same question should give the same result.

## 1. Before the professor arrives (15 minutes before)

1. Close anything that uses the GPU (another Python session, `chat_cli.py`, games). Browser tabs are fine.
2. Open PowerShell and run:

   ```powershell
   cd C:\Users\RRH\Desktop\VSD\truthlens-ai\TruthLensAI
   conda deactivate
   .venv\Scripts\Activate.ps1
   streamlit run frontend\streamlit_app.py
   ```

   A browser tab opens at `http://localhost:8501`. Leave the PowerShell window open; closing it stops the app.
3. **Warm-up:** type `What is football?` and click **Ask**. The first question loads Qwen, the LoRA adapter, the embedder and the checker (about 60 seconds). After that, questions take 5-30 seconds.
4. In the sidebar, keep **LoRA fine-tuned generator** and **Self-correction** switched ON.
5. Open the GitHub repo README in a second tab as backup.

## 2. One-sentence opener

> "TruthLens answers questions from 141 AI research papers, then splits its own answer into individual claims, checks each claim against the paper it cites, and gives a Trust Score. If a claim fails, it repairs the answer or declines to answer."

## 3. The demo, in order

### Demo 1: a clean, verified answer

**Ask:** `What is the difference between intrinsic and extrinsic hallucinations?`

**You will see:** **Trust Score: 95/100 High**, "2/2 claims supported". The whole answer is highlighted green.

**Show:**
- The **Claim verification** table: the answer was split into 2 claims, each checked against Source 1, with the checker's reason.
- Open **Sources → [1]** and point to the sentence in the paper that supports it.

**Say:** "Every highlighted sentence was checked against the passage it cites. Green means the cited source actually says it."

### Demo 2: what fine-tuning changed (base vs LoRA)

**Step 1:** switch **LoRA fine-tuned generator** OFF. **Ask:** `What causes hallucinations in large language models?`

**You will see:** a long answer, 6 claims, 98/100. One sentence is **yellow** ("Supported, uncited"): it has no citation.

**Step 2:** switch LoRA back ON and ask the same question.

**You will see:** one short sentence, fully cited, **100/100**.

**Say:** "Base Qwen writes longer answers and leaves 25-40% of sentences uncited. After QLoRA fine-tuning on 321 examples, uncited sentences went to 0% and correctly cited claims went from 59% to 85%. The trade-off is that answers got much shorter."

**Honest point (say it before you're asked):** the yellow sentence ("intrinsic hallucinations arise from the model itself...") is actually a wrong definition, and the checker still accepted it. This is the checker's known weakness: it catches only 4 of 10 unsupported claims, so a High score means "nothing was flagged", not "proven correct".

### Demo 3 (the highlight): a false-premise question

**Ask:** `Why does LoRA increase the number of trainable parameters compared to full fine-tuning?`
(The premise is false: LoRA *reduces* trainable parameters.)

**You will see:** the system **declines to answer**. Open the **Self-correction** section:
- **original, 39/100 Low:** the fine-tuned model accepted the false premise and wrote a confident, cited answer. The checker marked "LoRA increases the number of trainable parameters" as **Contradicted** by Source 2.
- **regenerate / prune:** both repairs ended in a refusal, which was kept.

**Say:** "This is our key finding. The fine-tuned model's mistakes look exactly like correct answers: fluent and properly cited. Citation format is not evidence. The verification layer caught the contradiction, and self-correction withheld the answer instead of showing it."

### Demo 4: a convincing citation the source doesn't back up

**Ask:** `What are the main challenges of deploying RAG in industry?`

**You will see:** declined. Under **Self-correction**, the original (24/100) said "data preprocessing, integration into existing workflows, and data privacy (Source 3)", but Source 3 does not say that.

**Say:** "The citation looks legitimate, but the cited passage doesn't contain those challenges. The system prefers saying 'I can't verify this' to inventing an answer."

**If asked why the score is 5/100 on a refusal:** for refusals the score measures *how justified declining is*. Here the sources were highly relevant (relevance 0.95), so a refusal is suspicious: it means "relevant papers exist, but no verifiable answer could be produced; a human should look". Compare with Demo 5.

### Demo 5: a question outside the corpus

**Ask:** `What is quantum entanglement?`

**You will see:** declined, **Trust Score: 93/100 High**, retrieval relevance 0.07.

The passages are folded away under "Closest passages found; none judged relevant". (In Demo 4 the same section reads "Passages retrieved; no verifiable answer could be built from them", because those passages *were* on-topic.)

**Say:** "Nothing relevant was retrieved, so declining is fully justified. The no-RAG baseline (Stage 1) answered this from memory, and on 'How many parameters does GPT-5 have?' it even called GPT-4 an Anthropic model."

### Finish: the Results page

Click **Results** in the sidebar and walk through the three tables:
1. **Stage 2 vs 3:** fine-tuning improved attribution (59% → 85% correctly cited) but not grounding (confirmed unsupported claims 1.3% vs 6.1%).
2. **Trust score:** the old heuristic scored the base model's bad answers *higher* than its good ones (77 vs 61); the claim-based score separates clean LoRA answers (91) from flawed ones (25.5).
3. **Self-correction:** on the 7 flagged answers, all 6 confirmed unsupported claims were removed and the 3 correct answers were kept; the cost is 3 answerable questions now declined.

## 4. Questions the professor is likely to ask

| Question | Answer |
|---|---|
| How do you know the checker is right? | I hand-labelled 78 claims, including a 40-claim set labelled blind: the checker got 36/40. It approves 94% of truly supported claims but catches only 4 of 10 unsupported ones, so all automated hallucination rates are lower bounds. |
| Why not NLI models for verification? | Tried first: an NLI cross-encoder approved only 15 of 30 truly supported claims. Then Qwen + NLI veto: an audit showed 10 of 13 vetoes were wrong and biased against the base model. Both designs are documented. |
| Isn't Qwen checking its own answers? | The checker runs with the LoRA adapter switched off, but it is the same model family, so shared blind spots are possible. Its leniency (4/10) is the measured symptom. A stronger independent judge is the top future upgrade. |
| Did fine-tuning reduce hallucination? | No. It improved citation accuracy, but after hand-checking, the fine-tuned model had more confirmed unsupported claims (6 of 99 vs 3 of 234). Small numbers, so weak evidence, but no evidence of a reduction. |
| Is the Trust Score a probability? | No. It is a heuristic: 60% claim support, 15% citation precision, 25% retrieval relevance, with caps (an unsupported claim → at most 69, a contradiction → at most 39). Calibration is untested; that's why it is never written as "% confident". |
| Why does self-correction refuse instead of fixing? | When told which claims failed, the model either declined or repeated the error (it fixed 0 of 4 real errors by regenerating). It can't supply a fact that isn't in the sources, only withhold it. |
| How big is the evaluation? | 50 items (12 fixed questions + 38 from 14 papers never seen in training), single annotator. Differences of 1-2 items are not meaningful, and no significance is claimed. |
| What's left? | RAGAS/DeepEval evaluation, FastAPI + Docker deployment, trust score calibration, and a retrain with longer answers and false-premise examples. |
| Why 4-bit and local? | The laptop GPU has 8 GB; 4-bit Qwen2.5-7B uses about 5.7 GB. Training ran on a free Colab T4. |

## 5. If something goes wrong

- **"CUDA out of memory" or the app hangs on the first question:** another program holds the GPU. Close it, press `Ctrl+C` in PowerShell, and run `streamlit run frontend\streamlit_app.py` again.
- **`torch.cuda.is_available()` is False / bitsandbytes error:** conda is active. Run `conda deactivate`, then `.venv\Scripts\Activate.ps1` again.
- **Browser tab closed:** reopen `http://localhost:8501`; the models stay loaded.
- **Total failure:** the **Results** page needs no GPU, and the GitHub README has every result. As a fallback, `python scripts\chat_cli.py` runs the same pipeline in the terminal.
- Take screenshots of Demos 1-5 the day before, so you can show them if the live demo fails.
