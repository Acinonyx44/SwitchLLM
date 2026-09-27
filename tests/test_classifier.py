import pytest

from switchllm import classify


@pytest.mark.parametrize(
    "prompt, task_class",
    [
        ("Quick tl;dr of this thread", "summarize"),
        ("Fix the bug in this python function", "code"),
        ("Review the indemnification clause in this contract", "legal"),
        ("Prove the theorem for all n", "math"),
        ("Compare the trade-offs of these two strategies", "analysis"),
        ("Draft an email to the client", "writing"),
        ("hello there", "chat"),
    ],
)
def test_task_class(prompt, task_class):
    assert classify(prompt).task_class == task_class


def test_difficulty_moves_with_hard_and_easy_signals():
    base = classify("Fix the bug in this python function").difficulty
    assert classify("Fix the bug in this python function, handle edge cases and race conditions").difficulty > base
    assert classify("Quick fix for a typo bug in this python function").difficulty < base


def test_difficulty_is_clamped():
    p = classify("prove, optimize, architect from scratch, rigorous, step by step? why? how? what?")
    assert 1 <= p.difficulty <= 5
    assert classify("quick short simple hi").difficulty == 1


def test_needs_tools():
    assert classify("Search the web for the latest news on rates").needs_tools
    assert not classify("Summarize this paragraph").needs_tools


def test_long_input_raises_difficulty():
    short = classify("Summarize this document.")
    long = classify("Summarize this document. " + "word " * 2000)
    assert long.difficulty == short.difficulty + 1
