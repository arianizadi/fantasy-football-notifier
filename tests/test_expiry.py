from notifier.expiry import expire_due_messages
from notifier.history import MessageHistory


def test_failed_deletion_remains_persisted_for_retry(tmp_path) -> None:
    path = tmp_path / "sent-messages.json"
    store = MessageHistory(path)
    store.record(123)

    result = expire_due_messages(store, ttl_seconds=0, delete=lambda _: False)

    assert result.failed == [123]
    assert result.deleted == []
    assert len(store) == 1
    assert len(MessageHistory(path)) == 1


def test_successful_deletion_is_forgotten(tmp_path) -> None:
    path = tmp_path / "sent-messages.json"
    store = MessageHistory(path)
    store.record(456)

    result = expire_due_messages(store, ttl_seconds=0, delete=lambda _: True)

    assert result.deleted == [456]
    assert result.failed == []
    assert len(store) == 0
    assert len(MessageHistory(path)) == 0
