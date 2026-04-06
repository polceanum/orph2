from llm_agent.eval import exact_match, normalize_text


def test_normalize_text_collapses_whitespace_and_case() -> None:
    assert normalize_text("  HeLLo\n\tWORLD  ") == "hello world"


def test_exact_match_ignores_case_and_whitespace() -> None:
    assert exact_match("  42\n", "42")
    assert exact_match("A B", "a    b")
    assert not exact_match("41", "42")
