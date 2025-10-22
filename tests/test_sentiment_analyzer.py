import pytest

from sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> SentimentAnalyzer:
    """Create a SentimentAnalyzer for tests.

    Returns:
        SentimentAnalyzer: Analyzer with deterministic config.
    """
    # Keep OpenAI disabled by default; tests will monkeypatch as needed.
    # Note that OpenAI is not cheap
    return SentimentAnalyzer(neutrality_band=0.2, emoji_nudge=0.1, use_openai=False)


@pytest.mark.parametrize(
    "text,stub_score,stub_label,expected_score,expected_label",
    [
        # Given a clearly positive stub result with no emojis
        # When analyze is called
        # Then the output reflects the Comprehend-based score/label (no emoji adjustment)
        ("Great product", 0.8, "POSITIVE", 0.8, "POSITIVE"),
        # Given a clearly negative stub result with no emojis
        ("Terrible experience", -0.7, "NEGATIVE", -0.7, "NEGATIVE"),
    ],
)
def test_basic_flow_uses_comprehend_result(
    analyzer: SentimentAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    stub_score: float,
    stub_label: str,
    expected_score: float,
    expected_label: str,
) -> None:
    """Basic flow returns the (score, label) derived from Comprehend.

    Given:
        A stubbed Comprehend response for the input text.
    When:
        Calling analyze(text).
    Then:
        The score/label match the stub (bounded/rounded), with no emoji influence.
    """
    monkeypatch.setattr(analyzer, "_analyze_with_comprehend", lambda _t: (stub_score, stub_label))
    score, label = analyzer.analyze(text)
    assert score == pytest.approx(expected_score, abs=1e-9)
    assert label == expected_label


@pytest.mark.parametrize(
    "base_text,emoji,trend",
    [
        # Given a neutral base score
        # When adding a positive emoji
        # Then score nudges positive (bounded by emoji_nudge)
        ("Just okay", "😍", "more_positive"),
        # Given a neutral base score
        # When adding a negative emoji
        # Then score nudges negative (bounded by emoji_nudge)
        ("Just okay", "😡", "more_negative"),
        # Given a neutral base score
        # When adding no emoji
        # Then score remains essentially unchanged
        ("Just okay", "", "no_change"),
    ],
)
def test_emojis_nudge_score(
    analyzer: SentimentAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
    base_text: str,
    emoji: str,
    trend: str,
) -> None:
    """Emoji presence nudges the score in the expected direction.

    Given:
        Comprehend returns a neutral baseline (0.0, "NEUTRAL").
    When:
        An emoji is appended to the text.
    Then:
        The score shifts toward the emoji polarity, bounded by emoji_nudge.
    """
    monkeypatch.setattr(analyzer, "_analyze_with_comprehend", lambda _t: (0.0, "NEUTRAL"))
    base_score, _ = analyzer.analyze(base_text)
    new_score, _ = analyzer.analyze(f"{base_text} {emoji}".strip())
    delta = new_score - base_score

    if trend == "more_positive":
        assert delta >= 0
        assert delta <= analyzer._emoji_nudge + 1e-9
    elif trend == "more_negative":
        assert delta <= 0
        assert abs(delta) <= analyzer._emoji_nudge + 1e-9
    else:
        assert abs(delta) < 0.05


@pytest.mark.parametrize("neutrality_band", [0.1, 0.2, 0.3])
def test_neutrality_band_controls_label(
    analyzer: SentimentAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
    neutrality_band: float,
) -> None:
    """Neutrality band determines whether borderline scores become NEUTRAL.

    Given:
        A borderline positive score from Comprehend (0.15, "POSITIVE").
    When:
        Adjusting the neutrality_band.
    Then:
        Label is NEUTRAL iff abs(score) < neutrality_band.
    """
    monkeypatch.setattr(analyzer, "_analyze_with_comprehend", lambda _t: (0.15, "POSITIVE"))
    analyzer._neutrality_band = neutrality_band  # test the decision boundary only
    score, label = analyzer.analyze("Borderline sentiment")
    assert (abs(score) < neutrality_band) == (label == "NEUTRAL")


def test_invalid_input_raises(analyzer: SentimentAnalyzer) -> None:
    """Invalid inputs should raise ValueError.

    Given:
        Empty or non-string inputs.
    When:
        Calling analyze.
    Then:
        A ValueError is raised.
    """
    with pytest.raises(ValueError):
        analyzer.analyze("   ")
    with pytest.raises(ValueError):
        analyzer.analyze(None)  # type: ignore[arg-type]


def test_openai_adjustment_average(
    analyzer: SentimentAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI adjustment averages scores when OpenAI is available.

    Given:
        Comprehend returns (0.6, POSITIVE) and OpenAI returns (0.0, NEUTRAL).
    When:
        OpenAI is enabled (stubbed) and analyze is called.
    Then:
        Final score is the average (0.3), label reflects neutrality band.
    """
    monkeypatch.setattr(analyzer, "_analyze_with_comprehend", lambda _t: (0.6, "POSITIVE"))
    monkeypatch.setattr(analyzer, "_analyze_with_openai", lambda _t: (0.0, "NEUTRAL"))
    analyzer._openai = object()  # mark OpenAI as "configured" without real client

    score, label = analyzer.analyze("Mixed-ish")
    assert score == pytest.approx(0.3, abs=1e-9)
    assert label == "POSITIVE"  # 0.3 > neutrality_band=0.2
