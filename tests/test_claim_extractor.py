from app.verification.claim_extractor import (
    is_refusal_answer,
    parse_citations,
    split_sentences,
    strip_citations,
)


def test_parse_citations_formats():
    assert parse_citations("X holds (Source 2).") == [2]
    assert parse_citations("X holds (Source 1, Source 3, Source 4).") == [1, 3, 4]
    assert parse_citations("As shown in Sources 2 and 5, X holds.") == [2, 5]
    assert parse_citations("According to Source 1, X holds.") == [1]
    assert parse_citations("No citation here.") == []


def test_strip_citations():
    assert strip_citations("LoRA freezes weights (Source 3).") == "LoRA freezes weights."
    assert strip_citations("According to Source 1, intrinsic errors contradict the input.") == "intrinsic errors contradict the input."
    assert strip_citations("1. **Vision-Language Understanding (VLU)**: models for comprehension.") == \
        "Vision-Language Understanding (VLU): models for comprehension."
    assert strip_citations("Extrinsic hallucinations, as defined in Source 1, refer to unverifiable content.") == \
        "Extrinsic hallucinations refer to unverifiable content."
    assert strip_citations("The method is evaluated in Sources 2 and 3 on GLUE.") == "The method is evaluated on GLUE."
    # parenthetical content that is not a citation is kept
    assert strip_citations("Mean reciprocal rank (MRR) is used (Source 5).") == "Mean reciprocal rank (MRR) is used."


def test_split_sentences_handles_lists_and_paragraphs():
    answer = "The categories are:\n\n1. **A**: first thing.\n2. **B**: second thing.\n\nThese help. Next one (Source 2)."
    assert split_sentences(answer) == [
        "The categories are:",
        "1. **A**: first thing.",
        "2. **B**: second thing.",
        "These help.",
        "Next one (Source 2).",
    ]


def test_split_does_not_break_on_abbreviations_followed_by_lowercase():
    assert len(split_sentences("Models e.g. BERT are used. They work (Source 1).")) == 2


def test_refusal_detection_ignores_cited_limitation_statements():
    assert is_refusal_answer("The provided sources do not contain information to answer this question.")
    assert not is_refusal_answer(
        "LLMs cannot answer questions about events that occurred after they were trained (Source 1)."
    )
