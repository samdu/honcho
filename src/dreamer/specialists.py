"""
Agentic specialists for the dream cycle.

Each specialist is a fully autonomous agent that:
1. Receives probing questions as entry points
2. Uses tools to search for relevant observations
3. Creates new observations (deductive or inductive)
4. Can delete duplicates (deduction only)
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import crud, schemas
from src.config import ConfiguredModelSettings, settings
from src.dependencies import tracked_db
from src.exceptions import ValidationException
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.prompts import load_template
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import DreamSpecialistEvent, emit
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import TokenTypes
from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    create_tool_executor,
)

logger = logging.getLogger(__name__)


def _require_specialist_model_config(
    model_config: ConfiguredModelSettings | None,
    *,
    specialist_name: str,
) -> ConfiguredModelSettings:
    if model_config is None:
        raise ValidationException(
            f"{specialist_name} MODEL_CONFIG must be resolved before use"
        )
    return model_config


@dataclass
class SpecialistResult:
    """Result of a specialist run for telemetry and aggregation."""

    run_id: str
    specialist_type: str
    iterations: int
    tool_calls_count: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    success: bool
    content: str


# Tool names to exclude when peer card creation is disabled
PEER_CARD_TOOL_NAMES = {"update_peer_card"}


class BaseSpecialist(ABC):
    """Base class for agentic specialists."""

    name: str = "base"
    # Subclasses can override to customize the peer card update instruction
    peer_card_update_instruction: str = (
        "Only update this with durable profile facts via `update_peer_card`."
    )

    @abstractmethod
    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        """Get the tools available to this specialist."""
        ...

    @abstractmethod
    def get_model_config(self) -> ConfiguredModelSettings:
        """Get the configured model to use for this specialist."""
        ...

    def get_max_tokens(self) -> int:
        """Get max output tokens for this specialist."""
        return 16384

    def get_max_iterations(self) -> int:
        """Get max tool iterations."""
        return 15

    @abstractmethod
    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        """Build the system prompt for this specialist."""
        ...

    @abstractmethod
    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        """Build the user prompt with optional exploration hints and current peer card."""
        ...

    def _build_peer_card_context(self, peer_card: list[str] | None) -> str:
        """Build the peer card context section for user prompts."""
        if not peer_card:
            return ""
        facts = "\n".join(f"- {fact}" for fact in peer_card)
        return f"""
## CURRENT PEER CARD

{facts}

{self.peer_card_update_instruction}
If you update it, send the full deduplicated list and remove stale entries.

"""

    async def run(
        self,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str | None,
        hints: list[str] | None = None,
        configuration: ResolvedConfiguration | None = None,
        parent_run_id: str | None = None,
    ) -> SpecialistResult:
        """
        Run the specialist agent.

        Uses short-lived DB sessions to avoid holding connections during LLM calls.

        Args:
            workspace_name: Workspace identifier
            observer: The observing peer
            observed: The peer being observed
            session_name: Session identifier
            hints: Optional hints to guide exploration (specialists explore freely if None)
            configuration: Resolved configuration for checking feature flags (optional)
            parent_run_id: Optional run_id from orchestrator for correlation

        Returns:
            SpecialistResult with metrics and content
        """
        run_id = parent_run_id or str(uuid.uuid4())[:8]
        task_name = f"dreamer_{self.name}_{run_id}"
        start_time = time.perf_counter()

        # Short-lived DB session for preflight operations
        async with tracked_db("dream.specialist.preflight") as db:
            await crud.get_peer(db, workspace_name, schemas.PeerCreate(name=observer))
            if observer != observed:
                await crud.get_peer(
                    db, workspace_name, schemas.PeerCreate(name=observed)
                )

            # Determine if peer card tools should be included
            peer_card_enabled = configuration is None or configuration.peer_card.create

            # Fetch current peer card to inject into prompt (saves a tool call)
            current_peer_card: list[str] | None = None
            if peer_card_enabled:
                current_peer_card = await crud.get_peer_card(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                )
        # DB session closed — LLM calls happen without holding a connection

        # Build messages
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    observed, peer_card_enabled=peer_card_enabled
                ),
            },
            {
                "role": "user",
                "content": self.build_user_prompt(hints, current_peer_card),
            },
        ]

        # Create tool executor with telemetry context
        tool_executor: Callable[
            [str, dict[str, Any]], Any
        ] = await create_tool_executor(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_observation_ids=True,
            history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
            configuration=configuration,
            run_id=run_id,
            agent_type=self.name,
            parent_category="dream",
        )

        model_config = self.get_model_config()

        # Respect operator-configured max_output_tokens on the specialist's
        # ModelConfig (e.g. DREAM_DEDUCTION_MODEL_CONFIG__MAX_OUTPUT_TOKENS).
        # Only fall back to the specialist's hardcoded default when the
        # config leaves max_output_tokens unset or non-positive.
        configured_max = model_config.max_output_tokens
        effective_max_tokens = (
            configured_max
            if configured_max and configured_max > 0
            else self.get_max_tokens()
        )

        # Track iterations via callback
        iteration_count = 0

        def iteration_callback(data: Any) -> None:
            nonlocal iteration_count
            iteration_count = data.iteration

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            model_config=model_config,
            prompt="",  # Ignored since we pass messages
            max_tokens=effective_max_tokens,
            tools=self.get_tools(peer_card_enabled=peer_card_enabled),
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=self.get_max_iterations(),
            messages=messages,
            track_name=f"Dreamer/{self.name}",
            iteration_callback=iteration_callback,
        )

        # Log metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", duration_ms, "ms")
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        accumulate_metric(task_name, "input_tokens", response.input_tokens, "count")
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "count")

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_dreamer_tokens(
                count=response.input_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.INPUT.value,
            )
            prometheus_metrics.record_dreamer_tokens(
                count=response.output_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.OUTPUT.value,
            )

        logger.info(
            f"{self.name}: Completed in {duration_ms:.0f}ms, "
            + f"{len(response.tool_calls_made)} tool calls, "
            + f"{response.input_tokens} in / {response.output_tokens} out"
        )

        log_performance_metrics(f"dreamer_{self.name}", run_id)

        # Emit telemetry event
        emit(
            DreamSpecialistEvent(
                run_id=run_id,
                specialist_type=self.name,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                iterations=iteration_count,
                tool_calls_count=len(response.tool_calls_made),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_ms=duration_ms,
                success=True,
            )
        )

        return SpecialistResult(
            run_id=run_id,
            specialist_type=self.name,
            iterations=iteration_count,
            tool_calls_count=len(response.tool_calls_made),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            success=True,
            content=response.content,
        )


class DeductionSpecialist(BaseSpecialist):
    """
    Creates deductive observations from explicit observations.

    This specialist:
    1. Explores recent observations and messages to understand what's there
    2. Identifies logical implications, knowledge updates, and contradictions
    3. Creates new deductive observations with premise linkage
    4. Deletes outdated observations
    5. Updates peer card with biographical facts
    """

    name: str = "deduction"
    peer_card_update_instruction: str = "Update this with `update_peer_card` only for stable biographical/profile facts."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return DEDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in DEDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model_config(self) -> ConfiguredModelSettings:
        return _require_specialist_model_config(
            settings.DREAM.DEDUCTION_MODEL_CONFIG,
            specialist_name="DREAM DEDUCTION",
        )

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 12

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = (
            load_template("dreamer_deduction_peer_card.md") if peer_card_enabled else ""
        )
        return load_template("dreamer_deduction_system.md").format(
            observed=observed, peer_card_section=peer_card_section
        )

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return load_template("dreamer_deduction_user_with_hints.md").format(
                peer_card_context=peer_card_context, hints_str=hints_str
            )

        return load_template("dreamer_deduction_user_freeform.md").format(
            peer_card_context=peer_card_context
        )


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Explores observations to understand what's there
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage
    4. Updates peer card with high-confidence traits and tendencies
    """

    name: str = "induction"
    peer_card_update_instruction: str = "Only add highly stable profile traits/preferences; do not copy transient conclusions."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return INDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in INDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model_config(self) -> ConfiguredModelSettings:
        return _require_specialist_model_config(
            settings.DREAM.INDUCTION_MODEL_CONFIG,
            specialist_name="DREAM INDUCTION",
        )

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 10

    def build_system_prompt(
        self, observed: str, *, peer_card_enabled: bool = True
    ) -> str:
        peer_card_section = (
            load_template("dreamer_induction_peer_card.md") if peer_card_enabled else ""
        )
        return load_template("dreamer_induction_system.md").format(
            observed=observed, peer_card_section=peer_card_section
        )

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return load_template("dreamer_induction_user_with_hints.md").format(
                peer_card_context=peer_card_context, hints_str=hints_str
            )

        return load_template("dreamer_induction_user_freeform.md").format(
            peer_card_context=peer_card_context
        )


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}
