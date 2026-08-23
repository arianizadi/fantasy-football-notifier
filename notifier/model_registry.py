"""Discover free models on OpenRouter at runtime.

Stealth models are unreleased models published under a codename, free while
the lab's promotional window is open, then withdrawn or renamed. Pinning one
by id guarantees a dead config the week it disappears, so candidates are
discovered from the live model list and filtered by actual price.

Free is verified from the pricing fields, never assumed from the namespace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from .logging_utils import structured_log

MODELS_URL = "https://openrouter.ai/api/v1/models"
REQUEST_TIMEOUT = 25
STEALTH_PREFIX = "stealth/"
# Structured output is non-negotiable: a candidate classifier must return JSON.
REQUIRED_PARAMS = {"response_format"}


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    name: str
    prompt_cost: float
    completion_cost: float
    context_length: int
    supported: frozenset[str]
    input_modalities: frozenset[str]
    output_modalities: frozenset[str]

    @property
    def is_text_generator(self) -> bool:
        """Accepts text and emits text only.

        Music and image models advertise `response_format` too, so filtering
        on supported parameters alone lets them through - an evaluation run
        wasted eight calls each on two Lyria audio models before this existed.
        """
        return "text" in self.input_modalities and self.output_modalities == {"text"}

    @property
    def is_free(self) -> bool:
        return self.prompt_cost == 0.0 and self.completion_cost == 0.0

    @property
    def is_stealth(self) -> bool:
        return self.model_id.startswith(STEALTH_PREFIX)

    @property
    def supports_structured_output(self) -> bool:
        return REQUIRED_PARAMS.issubset(self.supported)


def _parse(record: dict) -> ModelInfo | None:
    pricing = record.get("pricing") or {}
    try:
        prompt = float(pricing.get("prompt", "1"))
        completion = float(pricing.get("completion", "1"))
    except (TypeError, ValueError):
        return None
    architecture = record.get("architecture") or {}
    return ModelInfo(
        model_id=str(record.get("id") or ""),
        name=str(record.get("name") or ""),
        prompt_cost=prompt,
        completion_cost=completion,
        context_length=int(record.get("context_length") or 0),
        supported=frozenset(record.get("supported_parameters") or []),
        input_modalities=frozenset(architecture.get("input_modalities") or ["text"]),
        output_modalities=frozenset(architecture.get("output_modalities") or ["text"]),
    )


def list_models(session: requests.Session) -> list[ModelInfo]:
    response = session.get(MODELS_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    models = []
    for record in response.json().get("data", []):
        info = _parse(record)
        if info and info.model_id:
            models.append(info)
    return models


def free_candidates(
    session: requests.Session, *, stealth_only: bool = False
) -> list[ModelInfo]:
    """Free models that can return structured output, best context first."""
    candidates = [
        model
        for model in list_models(session)
        if model.is_free
        and model.is_text_generator
        and model.supports_structured_output
        and (model.is_stealth or not stealth_only)
    ]
    candidates.sort(key=lambda m: (not m.is_stealth, -m.context_length))
    structured_log(
        logging.INFO,
        "registry.free_candidates",
        count=len(candidates),
        stealthOnly=stealth_only,
        models=[m.model_id for m in candidates[:10]],
    )
    return candidates


def model_exists(session: requests.Session, model_id: str) -> bool:
    return any(m.model_id == model_id for m in list_models(session))
