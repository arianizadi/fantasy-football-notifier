from notifier.expiry import expire_due_messages
from notifier.history import MessageHistory
from unittest.mock import Mock


def test_native_retention_disables_custom_deletion(tmp_path) -> None:
    path = tmp_path / "sent-messages.json"
    store = MessageHistory(path)
    store.record(123)
    delete = Mock(return_value=True)

    result = expire_due_messages(store, ttl_seconds=0, delete=delete)

    assert result.failed == []
    assert result.deleted == []
    assert result.too_old == []
    delete.assert_not_called()
    assert len(store) == 1


def test_legacy_positive_ttl_is_also_inert(tmp_path) -> None:
    path = tmp_path / "sent-messages.json"
    store = MessageHistory(path)
    store.record(456)
    delete = Mock(return_value=True)

    result = expire_due_messages(store, ttl_seconds=3600, delete=delete)

    assert result.deleted == []
    assert result.failed == []
    delete.assert_not_called()
    assert len(store) == 1
