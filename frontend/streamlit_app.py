"""
TruthLens AI - Streamlit demo.

Runs the full pipeline in-process (no API backend yet): LoRA RAG -> claim verification ->
trust score -> self-correction. Needs the GPU free (close scripts\\chat_cli.py first).

Without a CUDA GPU (e.g. Streamlit Community Cloud), or with TRUTHLENS_REPLAY=1, it runs in replay
mode: the Ask page shows runs recorded by scripts\\record_demo.py (data/demo/recorded_runs.json)
and nothing heavier than streamlit is imported.

Run from the project root:
    streamlit run frontend\\streamlit_app.py
"""
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import streamlit as st

st.set_page_config(page_title="TruthLens AI", layout="wide")

VERDICT_STYLE = {  # label, colour (translucent so it reads in light and dark themes)
    "supported": ("Supported", "rgba(46, 160, 67, 0.25)"),
    "misattributed": ("Wrong source cited", "rgba(210, 153, 34, 0.30)"),
    "uncited_supported": ("Supported, uncited", "rgba(210, 153, 34, 0.30)"),
    "unsupported": ("Unsupported", "rgba(218, 54, 51, 0.30)"),
    "contradicted": ("Contradicted", "rgba(218, 54, 51, 0.45)"),
}
LABEL_COLOUR = {"High": "#2ea043", "Moderate": "#d29922", "Low": "#da3633"}
EXAMPLES = [
    "What causes hallucinations in large language models?",
    "What is the difference between intrinsic and extrinsic hallucinations?",
    "What is quantum entanglement?",
]
RECORDINGS = ROOT / "data/demo/recorded_runs.json"


@st.cache_resource
def replay_mode() -> bool:
    if os.environ.get("TRUTHLENS_REPLAY") == "1":
        return True
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True


@st.cache_data
def load_recordings() -> dict:
    if not RECORDINGS.exists():
        return {"runs": {}, "questions": []}
    return json.loads(RECORDINGS.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading Qwen2.5-7B (4-bit) + LoRA adapter...")
def warm_up(use_adapter: bool) -> bool:
    from app.rag.truthlens import DEFAULT_ADAPTER, load_generator
    return load_generator(DEFAULT_ADAPTER if use_adapter else None)


def annotated_answer(sentences: list[dict]) -> str:
    """Highlight each sentence by the worst verdict among its claims (precomputed in candidate_to_dict)."""
    parts = []
    for s in sentences:
        text = html.escape(s["text"])
        if s["verdict"]:
            label, colour = VERDICT_STYLE[s["verdict"]]
            parts.append(f'<span title="{label}" style="background:{colour};border-radius:4px;padding:1px 3px">{text}</span>')
        else:
            parts.append(text)
    return '<div style="line-height:1.9">' + " ".join(parts) + "</div>"


def trust_badge(trust: dict) -> str:
    colour = LABEL_COLOUR[trust["label"]]
    return (f'<div style="border-left:6px solid {colour};padding:8px 14px;border-radius:4px;'
            f'background:rgba(127,127,127,0.08)"><span style="font-size:1.6em;font-weight:700">'
            f'Trust Score: {trust["score"]}/100</span> <span style="color:{colour};font-weight:600">'
            f'{trust["label"]}</span></div>')


def page_ask():
    st.title("TruthLens AI")
    st.caption("Answers from 141 arXiv AI/ML papers. Every claim is checked against the source it cites.")
    replay = replay_mode()

    with st.sidebar:
        st.subheader("Pipeline")
        use_adapter = st.toggle("LoRA fine-tuned generator", value=True)
        self_correct = st.toggle("Self-correction", value=True)
        st.markdown("**Legend**")
        for key in ("supported", "uncited_supported", "misattributed", "unsupported", "contradicted"):
            label, colour = VERDICT_STYLE[key]
            st.markdown(f'<span style="background:{colour};padding:1px 6px;border-radius:4px">{label}</span>',
                        unsafe_allow_html=True)

    if replay:
        rec = load_recordings()
        st.info("**Replay mode.** This server has no GPU, so it shows recorded runs of the full pipeline "
                f"(Qwen2.5-7B + LoRA, greedy decoding, recorded {rec.get('recorded_on', 'n/a')}). "
                "The sidebar switches select the matching recording. "
                "To ask your own questions, run the app locally on a CUDA GPU (see the README).")
        if not rec["questions"]:
            st.warning("No recordings found (data/demo/recorded_runs.json).")
            return
        question = st.selectbox("Choose a question", rec["questions"])
        run = rec["runs"].get(f"{question}|adapter={int(use_adapter)}|correct={int(self_correct)}")
        if run is None:
            st.warning("No recording for this question with these settings.")
            return
        render_result(run, use_adapter)
        return

    example = st.selectbox("Try an example", [""] + EXAMPLES)
    question = st.text_input("Ask a question", value=example)
    if not (st.button("Ask", type="primary") and question.strip()):
        return

    warm_up(use_adapter)
    from app.rag.truthlens import DEFAULT_ADAPTER, result_to_dict, run_truthlens
    with st.spinner("Retrieving, generating, verifying claims..."):
        result = run_truthlens(question.strip(), adapter_path=DEFAULT_ADAPTER if use_adapter else None,
                               self_correct=self_correct)
    render_result(result_to_dict(result), use_adapter)


def render_result(run: dict, use_adapter: bool):
    """Render a result_to_dict() view of a live run or a recording."""
    best = run["best"]
    t = best["trust"]

    st.markdown(trust_badge(t), unsafe_allow_html=True)
    if t["is_refusal"]:
        st.caption(f"The system declined to answer. Retrieval relevance {t['retrieval_relevance']:.2f}: "
                   "a refusal scores higher when the retrieved sources are weak.")
    else:
        st.caption(f"{t['num_supported']}/{t['num_claims']} claims supported by their cited source - "
                   f"support {t['support']:.2f}, citation precision {t['citation_precision']:.2f}, "
                   f"retrieval relevance {t['retrieval_relevance']:.2f}"
                   + (f" - capped: {t['cap_applied']}" if t["cap_applied"] else ""))

    st.subheader("Answer")
    if best["claims"]:
        st.markdown(annotated_answer(best["sentences"]), unsafe_allow_html=True)
    else:
        st.write(run["answer"])
    st.caption(f"Generator: {'LoRA fine-tuned' if run['adapter_used'] and use_adapter else 'base'} Qwen2.5-7B-Instruct")

    if best["claims"]:
        st.subheader("Claim verification")
        st.dataframe([{
            "Claim": c["claim"],
            "Cites": ", ".join(f"S{n}" for n in c["cited_sources"]) or "-",
            "Verdict": VERDICT_STYLE[c["verdict"]][0],
            "Checker's reason": c["judge_reason"],
        } for c in best["claims"]], use_container_width=True, hide_index=True)

    candidates = run["candidates"]
    if len(candidates) > 1:
        st.subheader("Self-correction")
        st.write(f"The first answer had flagged claims, so {len(candidates) - 1} repairs were tried. "
                 f"Kept: **{best['strategy']}**.")
        for c in candidates:
            with st.expander(f"{c['strategy']} - Trust Score: {c['trust']['score']}/100 ({c['trust']['label']})",
                             expanded=c["strategy"] == "original"):
                st.write(c["answer"])

    sources = run["sources"]
    if not sources:
        return
    if t["is_refusal"]:
        # the answer used none of these, so keep them out of the way; wording depends on whether they were on-topic
        heading = ("Closest passages found; none judged relevant" if t["retrieval_relevance"] <= 0.5
                   else "Passages retrieved; no verifiable answer could be built from them")
        container = st.expander(heading, expanded=False)
    else:
        st.subheader("Sources")
        container = st.container()
    with container:
        for i, s in enumerate(sources, 1):
            label = f"[{i}] {s['title']} (arXiv:{s['arxiv_id']}), p. {s['page']} - similarity {s['similarity']:.3f}"
            if t["is_refusal"]:
                st.markdown(f"**{label}**")
                st.caption(f"Section: {s['section']}")
                st.write(s["text"])
            else:
                with st.expander(label):
                    st.caption(f"Section: {s['section']}")
                    st.write(s["text"])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def page_results():
    st.title("Evaluation results")
    st.caption("Fixed set (12 questions) + held-out set (38 items from 14 unseen papers). "
               "Small samples: differences of 1-2 items are not meaningful.")

    st.subheader("Stage 2 vs Stage 3: claim-level verification")
    st.dataframe({
        "Metric": ["Correctly cited claims", "Grounded (supported by any source)",
                   "Confirmed unsupported (manual check)", "Uncited claims"],
        "Base RAG": ["59.0%", "93.6%", "3 / 234 (1.3%)", "79"],
        "LoRA RAG": ["84.8%", "90.9%", "6 / 99 (6.1%)", "0"],
    }, use_container_width=True, hide_index=True)

    scores = read_jsonl(ROOT / "data/eval/trust_scores_phase8.jsonl")
    if scores:
        st.subheader("Trust score: interim vs claim-based")
        groups = []
        for system in ("base", "lora"):
            for name, keep in (("clean answers", lambda r: not r["is_refusal"] and not r["confirmed_unsupported"]),
                               ("answers with a confirmed unsupported claim", lambda r: not r["is_refusal"] and r["confirmed_unsupported"])):
                rs = [r for r in scores if r["system"] == system and keep(r)]
                if rs:
                    groups.append({"Group": f"{system}: {name}", "n": len(rs),
                                   "Interim score": round(sum(r["interim_score"] for r in rs) / len(rs), 1),
                                   "Claim-based score": round(sum(r["score"] for r in rs) / len(rs), 1)})
        st.dataframe(groups, use_container_width=True, hide_index=True)

    corrections = read_jsonl(ROOT / "data/eval/self_correction_phase8.jsonl")
    if corrections:
        st.subheader("Self-correction on flagged LoRA answers")
        st.dataframe([{
            "Item": r["item_id"], "Question": r["question"], "Kept": r["best"],
            **{c["strategy"]: c["score"] for c in r["candidates"]},
        } for r in corrections], use_container_width=True, hide_index=True)


page = st.sidebar.radio("Page", ["Ask", "Results"])
page_ask() if page == "Ask" else page_results()
