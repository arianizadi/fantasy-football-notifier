"""Process-local health timestamps exposed through Telegram ``/status``."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentHealth:
    ok: bool
    checked_at: float
    last_success_at: float
    detail: str = ""


class HealthRegistry:
    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._components: dict[str, ComponentHealth] = {}

    def mark(self, component: str, *, ok: bool, detail: str = "") -> None:
        now = time.time()
        with self._lock:
            previous = self._components.get(component)
            last_success = now if ok else (previous.last_success_at if previous else 0.0)
            self._components[component] = ComponentHealth(
                ok=ok,
                checked_at=now,
                last_success_at=last_success,
                detail=detail[:160],
            )

    def snapshot(self) -> dict[str, ComponentHealth]:
        with self._lock:
            return dict(self._components)


HEALTH = HealthRegistry()


def age_label(stamp: float, *, now: float | None = None) -> str:
    if stamp <= 0:
        return "never"
    seconds = max(0, int((time.time() if now is None else now) - stamp))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def duration_label(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
