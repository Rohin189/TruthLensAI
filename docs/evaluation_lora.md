# Stage 2 vs Stage 3: Base RAG vs LoRA-fine-tuned RAG

Generated: 2026-10-30T21:09:13.428962

Adapter: `models/lora_adapters/qwen25-7b-truthlens-lora`

Both systems receive identical retrieved sources (top_k=5, threshold 0.3) and decode greedily from the same 4-bit weights; only the LoRA adapter differs. Base RAG is re-run here rather than reusing `results_baseline.jsonl`, which used temperature 0.3 sampling.

Metrics are rule-based proxies (RAGAS/DeepEval come in Phase 12). `is_refusal` uses the interim `REFUSAL_RE`, which has known false positives (see `docs/trust_score_interim.md`).

## Fixed question set (12)

| Metric | Base RAG | LoRA RAG |
|---|---|---|
| behaviour_correct | 12/12 | 12/12 |
| trust_answerable_mean | 69.6 | 57.0 |
| trust_refusal_mean | 19.5 | 19.5 |
| uncited_sentence_rate | 0.252 | 0.0 |
| invalid_citations | 0 | 0 |
| mean_words | 107.0 | 22.833 |

## Held-out validation set (papers unseen in training)

| Metric | Base RAG | LoRA RAG |
|---|---|---|
| answered_when_answerable | 30/30 | 30/30 |
| cites_gold_source | 23/30 | 21/30 |
| cites_gold_paper | 29/30 | 30/30 |
| all_citations_gold | 10/30 | 21/30 |
| content_recall_mean | 0.717 | 0.647 |
| uncited_sentence_rate | 0.401 | 0.0 |
| refused_when_unanswerable | 7/8 | 6/8 |
| mean_words | 77.974 | 25.684 |

---

## Per-question answers

### q1: What causes hallucinations in large language models?

Expected: answer

**Base RAG** (Trust Score: 81/100 High; uncited sentences 1/6; correct behaviour)

> Hallucinations in large language models refer to the generation of statements that appear plausible but are fabricated or contradict factual knowledge (Source 1). These hallucinations can be categorized into intrinsic and extrinsic hallucinations (Source 1). Intrinsic hallucinations arise from the model itself, while extrinsic hallucinations might be influenced by external factors. The tendency of large language models to produce such hallucinations poses a challenge to their reliability in practical applications (Source 1, Source 3, Source 4).
>
> The sources indicate that hallucinations are a significant issue for large language models, affecting their utility in various tasks and potentially undermining their reliability (Source 1, Source 2, Source 3, Source 4). However, the specific causes of hallucinations are not detailed in the provided sources; they are generally described as a result of the model's inherent limitations in generating accurate and factual information (Source 1).

**LoRA RAG** (Trust Score: 60/100 Moderate; uncited sentences 0/1; correct behaviour)

> Hallucinations in large language models refer to the generation of statements that appear plausible but are fabricated or contradict factual knowledge (Source 1).

---

### q2: How does LoRA reduce the number of trainable parameters?

Expected: answer

**Base RAG** (Trust Score: 59/100 Moderate; uncited sentences 4/5; correct behaviour)

> LoRA reduces the number of trainable parameters by approximating a large weight matrix \( W \in \mathbb{R}^{d \times k} \) with the product of two smaller matrices \( A \in \mathbb{R}^{d \times r} \) and \( B \in \mathbb{R}^{r \times k} \), where \( r \ll \min(d, k) \). This approximation is expressed as \( W \approx A \cdot B \). During fine-tuning, only the matrices \( A \) and \( B \) are updated, while \( W \) remains fixed. This approach reduces the number of trainable parameters to \( r \times (d + k) \), which is much smaller than \( d \times k \). For example, if \( d = 512 \), \( k = 64 \), and \( r = 8 \), the reduction in parameters is from 32,768 to 4,608, as shown in Source 1.

**LoRA RAG** (Trust Score: 59/100 Moderate; uncited sentences 0/1; correct behaviour)

> LoRA reduces the number of trainable parameters by inserting additional low-rank trainable matrices into existing layers, rather than updating all original parameters (Source 5).

---

### q3: What is retrieval augmented generation?

Expected: answer

**Base RAG** (Trust Score: 75/100 High; uncited sentences 0/3; correct behaviour)

> Retrieval Augmented Generation (RAG) is a technique that enhances the capabilities of large language models (LLMs) by integrating external, non-parametric knowledge provided by retrievers (Source 4). This integration allows LLMs to dynamically access and utilize relevant external knowledge, significantly improving their ability to generate accurate, contextually grounded, and informative responses (Source 3). The retriever module accesses a knowledge base that can be updated and adapted for different domains, offering benefits like maintainability and domain-specific improvements (Source 5).

**LoRA RAG** (Trust Score: 61/100 Moderate; uncited sentences 0/1; correct behaviour)

> Retrieval augmented generation (RAG) is a method that enhances the capabilities of large language models (LLMs) by integrating external, non-parametric knowledge provided by retrievers (Source 4).

---

### q4: What is QLoRA and how does it differ from LoRA?

Expected: answer

**Base RAG** (Trust Score: 70/100 High; uncited sentences 1/7; correct behaviour)

> QLoRA (Quantized Low-Rank Adaptation) is an extension of LoRA (Low-Rank Adaptation). It combines low-rank adapters with the quantization of pretrained weight matrices (Source 3). The primary goal of QLoRA is to further reduce memory usage while retaining the representational power of large Transformer backbones, making fine-tuning feasible on hardware with resource constraints (Source 3).
>
> Compared to LoRA, QLoRA introduces additional overhead from quantization and dequantization during forward passes, which can slow down training slightly (Source 1). However, both methods offer substantial efficiency gains over full fine-tuning, reducing the number of trainable parameters and achieving faster training times (Source 1, Source 3).
>
> In terms of performance, QLoRA generally matches or closely approaches the performance of LoRA but may show slight underperformance in certain settings due to approximation errors introduced by quantization (Source 5). Additionally, recent evaluations indicate that at lower precision levels, including 2-bit quantization, LowRA (another variant) outperforms QLoRA and other sub-4-bit precision methods (Source 4).

**LoRA RAG** (Trust Score: 49/100 Moderate; uncited sentences 0/1; correct behaviour)

> Quantized Low-Rank Adaptation (QLoRA) combines low-rank adapters with quantization of the pretrained weight matrices, aiming to reduce memory usage while retaining the representational power of large Transformer backbones (Source 3).

---

### q5: How can hallucinations be detected in LLM outputs?

Expected: answer

**Base RAG** (Trust Score: 72/100 High; uncited sentences 2/5; correct behaviour)

> Hallucinations in LLM outputs can be detected through various methods. One approach involves using a black-box hallucination detection method, as described in Source 1, which evaluates the effectiveness of such a method on models like ChatGPT and Llama2 without requiring external resources. Another method leverages the fact that language models are trained to answer yes or no questions, as mentioned in Source 5, where an LLM is prompted to answer "yes" or "no" to questions about its own responses to detect inconsistencies or hallucinations. Additionally, Source 4 discusses the development of a unified benchmark for hallucination detection across diverse scenarios, highlighting the importance of detecting hallucinations in high-risk domains. Overall, the detection of hallucinations in LLM outputs is a critical area of research, with various methods being explored to improve the reliability and safety of these models.

**LoRA RAG** (Trust Score: 58/100 Moderate; uncited sentences 0/1; correct behaviour)

> Hallucinations can be detected through various methods, including black-box, gray-box, and white-box approaches (Source 4).

---

### q6: What are common chunking strategies for RAG systems?

Expected: answer

**Base RAG** (Trust Score: 58/100 Moderate; uncited sentences 3/9; correct behaviour)

> Common chunking strategies for RAG systems include:
>
> 1. **Fixed-size Chunking**: This strategy involves dividing documents into chunks of a fixed size. It is suitable for simple documents and FAQ systems due to its simplicity and low computational overhead (Source 1).
>
> 2. **Overlapping Chunking**: This approach creates overlapping chunks, which can be beneficial for technical documentation and legal texts where context preservation is crucial, though it comes with higher computational overhead (Source 1).
>
> 3. **Semantic Windowing**: This is a more complex strategy designed for research papers and complex narratives, where understanding the context over a broader range of text is important. It requires high computational resources (Source 1).
>
> 4. **Topic-based Segmentation**: This method segments documents based on topics, which can be useful for multi-topic documents but also comes with high computational overhead (Source 1).
>
> These strategies are chosen based on the specific needs of the use case, document format, and performance requirements (Source 1). The choice of chunking strategy can significantly impact the accuracy of responses and the quality of retrievals in RAG systems (Source 1).

**LoRA RAG** (Trust Score: 58/100 Moderate; uncited sentences 0/1; correct behaviour)

> Common chunking strategies include fixed-size chunking, overlapping chunking, semantic windowing, and topic-based segmentation (Source 1).

---

### q7: What is the difference between intrinsic and extrinsic hallucinations?

Expected: answer

**Base RAG** (Trust Score: 59/100 Moderate; uncited sentences 0/3; correct behaviour)

> Intrinsic hallucinations occur when the model generates content that synthesizes information present in the input, while extrinsic hallucinations involve model generations that ignore the source material altogether (Source 2). According to Source 1, intrinsic hallucinations happen when the LLM output contradicts the input content, such as generating a caption that includes details not present in the input image in a multi-modal image captioning task. Extrinsic hallucinations, as defined in Source 1, refer to generated content that cannot be verified based on the source content.

**LoRA RAG** (Trust Score: 52/100 Moderate; uncited sentences 0/1; correct behaviour)

> Intrinsic hallucinations occur when the LLM output contradicts the input content, while extrinsic hallucinations refer to the generated content that cannot be verified based on the source content (Source 1).

---

### q8: How does self-correction improve LLM response quality?

Expected: answer

**Base RAG** (Trust Score: 73/100 High; uncited sentences 1/4; correct behaviour)

> Self-correction improves LLM response quality through an iterative process where the model generates a critique of its initial response, identifies shortcomings, and then revises the response based on this self-critique (Source 2, Source 4). This process has shown promise in enhancing the output quality of LLMs (Pan et al., 2023). Additionally, self-correction can involve the use of external tools or knowledge to refine responses, potentially leading to higher quality outputs (Source 1). The effectiveness of self-correction is evident in various applications, including reducing harmful content, incorporating specific keywords, and debugging code (Source 2).

**LoRA RAG** (Trust Score: 59/100 Moderate; uncited sentences 0/1; correct behaviour)

> Self-correction improves LLM response quality by allowing the model to refine its outputs based on feedback, leading to more accurate and contextually appropriate responses (Source 2).

---

### q9: What evaluation metrics are used for RAG systems?

Expected: answer

**Base RAG** (Trust Score: 78/100 High; uncited sentences 3/7; correct behaviour)

> Evaluation metrics for RAG systems include both traditional information retrieval (IR) metrics and more specific RAG-specific considerations. Traditional IR metrics such as Precision@k and Recall@k are used to measure the relevance and coverage of relevant documents in the top results. Additionally, modern retrieval metrics incorporate RAG-specific aspects like context utilization and fragment attribution (Source 5).
>
> Moreover, the evaluation of RAG systems aims to balance the quality of assessment with the efficiency of automation. This involves using both traditional metrics for baseline performance and advanced language model-based judges for semantic evaluation (Source 2). The operational playbooks also recommend coupling offline benchmarks with online telemetry data (such as latency, click-through rate, and deflection rate) and human review queues for drift control (Source 4).
>
> These metrics and evaluation practices help assess the effectiveness of different components of the RAG system, including retrieval time, processing speed, and the extent to which specific requirements are met (Source 1).

**LoRA RAG** (Trust Score: 57/100 Moderate; uncited sentences 0/1; correct behaviour)

> RAG systems are evaluated using a combination of traditional information retrieval metrics like precision, recall, and MRR, along with more recent metrics that consider context utilization and fragment attribution (Source 5).

---

### q10: What are the main challenges of deploying RAG in industry?

Expected: answer

**Base RAG** (Trust Score: 71/100 High; uncited sentences 0/3; correct behaviour)

> One of the main challenges of deploying RAG in industry is data preprocessing for preparing the retrieval component of the system (Source 3). Additionally, while specific challenges are not detailed in the provided sources, the study indicates that companies face challenges that can lead to wasted resources if not properly addressed (Source 1). These challenges, along with data preprocessing, are important to understand for informing effective adoption strategies and implementation guidelines (Source 1, Source 4).

**LoRA RAG** (Trust Score: 57/100 Moderate; uncited sentences 0/1; correct behaviour)

> The main challenges include data preprocessing, integration into existing workflows, and ensuring data privacy and security (Source 3).

---

### q11: What is quantum entanglement?

Expected: refuse

**Base RAG** (Trust Score: 20/100 Low; uncited sentences 2/2; correct behaviour)

> The provided sources do not contain any information about quantum entanglement. Therefore, I cannot answer the question based on the given information.

**LoRA RAG** (Trust Score: 20/100 Low; uncited sentences 1/1; correct behaviour)

> I cannot answer this question based on the provided sources, as they do not contain the relevant information.

---

### q12: How many trainable parameters does GPT-5 have?

Expected: refuse

**Base RAG** (Trust Score: 19/100 Low; uncited sentences 2/2; correct behaviour)

> The provided sources do not contain specific information about the number of trainable parameters for GPT-5. Therefore, based on the given information, I cannot determine how many trainable parameters GPT-5 has.

**LoRA RAG** (Trust Score: 19/100 Low; uncited sentences 1/1; correct behaviour)

> I cannot answer this question based on the provided sources, as they do not contain the relevant information.

---
