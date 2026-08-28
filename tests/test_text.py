from verijoin.text import exact_match, find_span, normalize_answer, token_f1, token_prf


def test_official_style_answer_normalization() -> None:
    assert normalize_answer("The, Delta City!") == "delta city"
    assert exact_match("Delta City", ["the delta city"]) == 1.0
    assert token_f1("Delta", ["Delta City"]) == 2 / 3
    assert token_prf("Delta", ["Delta City"]) == (1.0, 0.5, 2 / 3)


def test_boolean_mismatch_has_zero_official_f1() -> None:
    assert token_f1("yes indeed", ["yes"]) == 0.0
    assert token_f1("no", ["yes"]) == 0.0


def test_span_is_literal_and_preserves_offsets() -> None:
    text = "Ada Writer was born in Delta City."
    start, end, value = find_span(text, ["Delta City"]) or (-1, -1, "")
    assert text[start:end] == value == "Delta City"
