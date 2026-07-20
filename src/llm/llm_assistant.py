"""
LLM Assistant — reasoning assistant using Ollama (local) or mock.

Responsibilities:
  - Describe transformations in natural language.
  - Generate hypotheses from text descriptions.
  - Explain failures.
  - Rank candidate programs.
  - Suggest new search directions.

The LLM is NEVER used to directly predict output grids.
All grid generation is done by the DSL executor.

Providers:
  - ollama: Local Ollama server (default for offline Kaggle execution)
  - mock:   Returns canned responses (no dependencies needed)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...


class MockProvider(LLMProvider):
    """Mock provider for testing / offline use."""

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        # Return a generic response that never breaks the pipeline
        return json.dumps({
            "hypotheses": ["Try rotating 90 degrees", "Try mirroring horizontally"],
            "explanation": "Mock LLM response — no local model available.",
            "ranking": [],
        })

    def is_available(self) -> bool:
        return True


class OllamaProvider(LLMProvider):
    """Provider that calls a local Ollama server."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 30,
    ) -> None:
        self.host = host
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import urllib.request, urllib.error
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama request failed: {e} — falling back to mock")
            return MockProvider().complete(prompt, max_tokens)

    def is_available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=3)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# LLM Assistant
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from the LLM assistant."""

    hypotheses: List[str] = field(default_factory=list)
    explanation: str = ""
    ranked_programs: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw: str = ""


class LLMAssistant:
    """High-level LLM reasoning assistant.

    Usage::

        assistant = LLMAssistant(provider="ollama")
        response = assistant.describe_transformation(pairs)
        response = assistant.explain_failure(program, result)
        response = assistant.rank_candidates(programs, pairs)
    """

    _SYSTEM_PREFIX = (
        "You are an expert at solving ARC (Abstraction and Reasoning Corpus) puzzles. "
        "Grids are 2D arrays of integers 0-9 representing colours. "
        "Provide concise, structured JSON responses."
    )

    def __init__(
        self,
        provider: str = "mock",
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 30,
    ) -> None:
        if provider == "ollama":
            self._provider: LLMProvider = OllamaProvider(host=host, model=model, timeout=timeout)
            if not self._provider.is_available():
                logger.warning("Ollama not available — falling back to mock")
                self._provider = MockProvider()
        else:
            self._provider = MockProvider()

        self._enabled = self._provider.is_available()
        logger.info(f"LLMAssistant: provider={provider!r}, enabled={self._enabled}")

    def describe_transformation(self, pairs: List[Dict]) -> LLMResponse:
        """Ask the LLM to describe the transformation rule.

        Args:
            pairs: Training pairs with "input"/"output" ArcGrids.

        Returns:
            LLMResponse with hypotheses list.
        """
        if not self._enabled:
            return LLMResponse()

        prompt = self._build_description_prompt(pairs)
        raw = self._provider.complete(prompt, max_tokens=256)
        return self._parse_response(raw)

    def explain_failure(self, program_repr: str, failure_desc: str) -> LLMResponse:
        """Ask the LLM to explain a verification failure.

        Args:
            program_repr: String representation of the failing program.
            failure_desc: Text description of the failure.

        Returns:
            LLMResponse with explanation and suggestions.
        """
        if not self._enabled:
            return LLMResponse()

        prompt = (
            f"{self._SYSTEM_PREFIX}\n\n"
            f"The following ARC DSL program failed:\n{program_repr}\n\n"
            f"Failure description: {failure_desc}\n\n"
            "Explain why this might fail and suggest 3 alternative approaches. "
            "Respond in JSON: {\"explanation\": \"...\", \"suggestions\": [...]}"
        )
        raw = self._provider.complete(prompt, max_tokens=300)
        return self._parse_response(raw)

    def rank_candidates(self, programs: List[str], task_description: str) -> LLMResponse:
        """Ask the LLM to rank candidate programs.

        Args:
            programs:         List of program string representations.
            task_description: Text description of the task.

        Returns:
            LLMResponse with ranked_programs.
        """
        if not self._enabled or not programs:
            return LLMResponse()

        prog_list = "\n".join(f"{i}. {p}" for i, p in enumerate(programs))
        prompt = (
            f"{self._SYSTEM_PREFIX}\n\n"
            f"Task: {task_description}\n\n"
            f"Candidate programs:\n{prog_list}\n\n"
            "Rank these programs from best to worst and explain why. "
            "Respond in JSON: {\"ranked_programs\": [\"0\", \"2\", \"1\"], \"explanation\": \"...\"}"
        )
        raw = self._provider.complete(prompt, max_tokens=300)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_description_prompt(self, pairs: List[Dict]) -> str:
        examples = []
        for i, pair in enumerate(pairs[:3]):  # limit context
            inp = pair["input"].to_list()
            out = pair["output"].to_list()
            examples.append(f"Example {i+1}:\n  Input: {inp}\n  Output: {out}")
        examples_str = "\n".join(examples)
        return (
            f"{self._SYSTEM_PREFIX}\n\n"
            f"Observe these ARC transformation examples:\n{examples_str}\n\n"
            "What transformation rule converts each input to its output? "
            "Respond in JSON: {\"hypotheses\": [\"...\", \"...\"], \"explanation\": \"...\"}"
        )

    def _parse_response(self, raw: str) -> LLMResponse:
        try:
            data = json.loads(raw)
        except Exception:
            # Try to extract JSON from within the response
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    data = {}
            else:
                data = {}

        return LLMResponse(
            hypotheses=data.get("hypotheses", []),
            explanation=data.get("explanation", ""),
            ranked_programs=data.get("ranked_programs", []),
            suggestions=data.get("suggestions", []),
            raw=raw,
        )


__all__ = ["LLMAssistant", "LLMResponse", "OllamaProvider", "MockProvider"]
