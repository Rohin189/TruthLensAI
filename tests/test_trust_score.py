from app.self_correction.corrector import PRUNED_REFUSAL, prune
from app.trust_score.trust_score import compute_claim_trust_score, retrieval_relevance
from app.verification.claim_extractor import split_sentences
from app.verification.verifier import AnswerVerification, ClaimVerdict


def verdict(text, v, idx, cited=(1,)):
    return ClaimVerdict(text, list(cited), v, v, "", "", 0.0, False, [], idx)


def test_relevance_rescales_bge_range():
    assert retrieval_relevance([0.45]) == 0.0
    assert retrieval_relevance([0.62, 0.65]) == 0.5
    assert retrieval_relevance([0.9]) == 1.0


def test_all_supported_high_and_unsupported_capped():
    sims = [0.82]
    good = compute_claim_trust_score(["supported"] * 3, [True] * 3, sims, False)
    assert good.score == 100 and good.label == "High"
    one_bad = compute_claim_trust_score(["supported"] * 9 + ["unsupported"], [True] * 10, sims, False)
    assert one_bad.score == 69 and one_bad.cap_applied == "unsupported claim"
    contra = compute_claim_trust_score(["supported", "contradicted"], [True, True], sims, False)
    assert contra.score <= 39 and contra.label == "Low"


def test_refusal_scores_justification():
    off_topic = compute_claim_trust_score([], [], [0.45], True)
    strong = compute_claim_trust_score([], [], [0.85], True)
    assert off_topic.score == 100 and strong.score == 0


def test_prune_drops_flagged_sentences_and_reindexes():
    answer = "A is true (Source 1). B is made up (Source 2). C is true (Source 1)."
    v = AnswerVerification(False, [verdict("A is true.", "supported", 0),
                                   verdict("B is made up.", "unsupported", 1),
                                   verdict("C is true.", "supported", 2)])
    text, pv = prune(answer, v)
    assert "B is made up" not in text
    sentences = split_sentences(text)
    assert [sentences[c.sentence_index].startswith(c.claim[0]) for c in pv.claims] == [True, True]


def test_prune_everything_becomes_refusal():
    v = AnswerVerification(False, [verdict("B is made up.", "unsupported", 0)])
    text, pv = prune("B is made up (Source 2).", v)
    assert text == PRUNED_REFUSAL and pv.is_refusal


def test_select_prefers_refusal_over_mostly_unverified_answer():
    from app.self_correction.corrector import Candidate, score, select
    sims = [0.75]
    bad = AnswerVerification(False, [verdict("X.", "unsupported", 0)])
    ok = AnswerVerification(False, [verdict("A.", "supported", 0), verdict("B.", "supported", 1),
                                    verdict("C.", "unsupported", 2)])
    refusal = AnswerVerification(True)
    cands = [Candidate("original", "x", bad, score(bad, sims)), Candidate("prune", "r", refusal, score(refusal, sims))]
    assert select(cands).strategy == "prune"
    cands = [Candidate("original", "abc", ok, score(ok, sims)), Candidate("prune", "r", refusal, score(refusal, sims))]
    assert select(cands).strategy == "original"
