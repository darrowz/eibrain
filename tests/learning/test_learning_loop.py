from __future__ import annotations


def test_learning_components_record_and_score_outcome() -> None:
    from eibrain.learning.adaptation import AdaptationEngine
    from eibrain.learning.evaluation import EvaluationEngine
    from eibrain.learning.review import SelfReviewEngine

    review = SelfReviewEngine()
    evaluation = EvaluationEngine()
    adaptation = AdaptationEngine()

    review_item = review.review_turn(
        event_type="audio_transcript_final",
        transcript="hello",
        reply="reply: hello",
        outcome="speech_playback_completed",
    )
    score = evaluation.score_review(review_item)
    decision = adaptation.decide(score)

    assert score >= 0.0
    assert decision in {"keep_policy", "adjust_policy"}
    assert "quality" in review_item
    assert "recommendation" in review_item
