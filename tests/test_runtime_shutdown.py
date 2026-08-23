import importlib.util
import signal
from pathlib import Path
from unittest.mock import Mock


def _load_runner_module():
    path = Path(__file__).resolve().parent.parent / "bin" / "run-notifier.py"
    spec = importlib.util.spec_from_file_location("fantasy_notifier_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sigterm_requests_graceful_notifier_stop(monkeypatch) -> None:
    module = _load_runner_module()
    notifier = Mock()
    previous = object()
    captured = {}

    def register(signum, handler):
        captured["signum"] = signum
        captured["handler"] = handler
        return previous

    monkeypatch.setattr(module.signal, "signal", register)
    monkeypatch.setattr(module, "structured_log", Mock())

    assert module._install_sigterm_handler(notifier) is previous
    assert captured["signum"] == signal.SIGTERM

    captured["handler"](signal.SIGTERM, None)

    notifier.request_stop.assert_called_once_with()
