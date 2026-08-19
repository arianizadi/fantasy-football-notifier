from notifier.verify import format_reply


def test_verifier_note_is_escaped_for_telegram_html() -> None:
    text = format_reply(4, 2, "Start A < B & avoid <script>alert(1)</script>")

    assert "<b>disagrees, 2/5</b>" in text
    assert "A &lt; B &amp; avoid &lt;script&gt;" in text
    assert "<script>" not in text
