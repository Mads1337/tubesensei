"""
InvestigationRunner service - Runs InvestigationAgent LLM calls against Ideas.

Fetches the agent and idea, builds the prompt, calls the LLM, and stores
the result in an InvestigationRun record with full status tracking and cost
metadata.
"""
import json
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm_manager import LLMManager, ModelType
from app.core.exceptions import NotFoundException
from app.models.idea import Idea
from app.models.investigation_agent import InvestigationAgent
from app.models.investigation_run import InvestigationRun, InvestigationRunStatus

logger = logging.getLogger(__name__)

# Mapping from config string values to ModelType enum
_MODEL_TYPE_MAP: dict[str, ModelType] = {
    "fast": ModelType.FAST,
    "balanced": ModelType.BALANCED,
    "quality": ModelType.QUALITY,
}


class InvestigationRunner:
    """Runs an InvestigationAgent against an Idea and stores the result."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_list_as_bullets(items: list | None) -> str:
        """Format a JSONB list into a bullet-point string, or a fallback message."""
        if not items:
            return "None identified"
        bullets = []
        for item in items:
            if isinstance(item, dict):
                # Use a 'text', 'name', or 'description' key if present, else stringify
                text = (
                    item.get("text")
                    or item.get("name")
                    or item.get("description")
                    or str(item)
                )
            else:
                text = str(item)
            bullets.append(f"- {text}")
        return "\n".join(bullets)

    @staticmethod
    def _build_user_prompt(template: str, idea: Idea) -> str:
        """
        Substitute all {idea_*} placeholders in *template* with values from
        *idea*.  Unknown placeholders are left as-is so they don't raise.
        """
        tags = list(idea.tags) if idea.tags else []  # type: ignore[arg-type]
        techs = list(idea.technologies) if idea.technologies else []  # type: ignore[arg-type]
        challenges = list(idea.potential_challenges) if idea.potential_challenges else []  # type: ignore[arg-type]
        monetization = list(idea.monetization_strategies) if idea.monetization_strategies else []  # type: ignore[arg-type]

        replacements = {
            "idea_title": idea.title or "",
            "idea_description": idea.description or "",
            "idea_category": idea.category or "Uncategorized",
            "idea_tags": ", ".join(tags) if tags else "None",
            "idea_technologies": ", ".join(techs) if techs else "None",
            "idea_confidence_score": str(idea.confidence_score),
            "idea_complexity_score": (
                str(idea.complexity_score) if idea.complexity_score is not None else "N/A"
            ),
            "idea_market_size": idea.market_size_estimate or "Unknown",
            "idea_target_audience": idea.target_audience or "Unknown",
            "idea_competitive_advantage": idea.competitive_advantage or "None specified",
            "idea_challenges": InvestigationRunner._format_list_as_bullets(challenges),
            "idea_monetization": InvestigationRunner._format_list_as_bullets(monetization),
        }
        # Use str.format_map with a defaultdict-like object so unknown keys are
        # kept verbatim rather than raising KeyError.
        class _SafeMap(dict):
            def __missing__(self, key: str) -> str:  # type: ignore[override]
                return "{" + key + "}"

        return template.format_map(_SafeMap(replacements))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_investigation(
        self, agent_id: UUID, idea_id: UUID, model_override: str | None = None
    ) -> InvestigationRun:
        """
        Run an investigation agent against an idea.

        Steps:
        1. Fetch the InvestigationAgent and Idea from the database.
        2. Create an InvestigationRun record with status PENDING.
        3. Transition to RUNNING and commit.
        4. Build the formatted user prompt.
        5. Call the LLM.
        6. Persist result content, token usage, and cost.
        7. Attempt to parse JSON from the result into result_structured.
        8. Set status COMPLETED and commit.
        9. On any error: set status FAILED with error_message and commit.
        """
        # Step 1 - fetch agent and idea
        agent = await self.db.get(InvestigationAgent, agent_id)
        if agent is None:
            raise NotFoundException("InvestigationAgent", str(agent_id))

        idea = await self.db.get(Idea, idea_id)
        if idea is None:
            raise NotFoundException("Idea", str(idea_id))

        # Step 2 - create run with PENDING status
        run = InvestigationRun(
            agent_id=agent_id,
            idea_id=idea_id,
            status=InvestigationRunStatus.PENDING,
        )
        self.db.add(run)
        await self.db.commit()
        await self.db.refresh(run)

        # Step 3 - transition to RUNNING
        run.status = InvestigationRunStatus.RUNNING
        await self.db.commit()

        try:
            # Step 4 - build the user prompt
            user_prompt = self._build_user_prompt(str(agent.user_prompt_template), idea)

            # Step 5 - resolve model settings from agent config
            model_type_str: str = (agent.config or {}).get("model_type", "balanced")
            model_type: ModelType = _MODEL_TYPE_MAP.get(
                model_type_str.lower(), ModelType.BALANCED
            )
            temperature = (agent.config or {}).get("temperature")
            max_tokens = (agent.config or {}).get("max_tokens")

            logger.info(
                "Running investigation: agent=%s idea=%s model_type=%s model_override=%s",
                agent_id,
                idea_id,
                model_type.value,
                model_override,
            )

            # Step 6 - call the LLM
            llm_manager = LLMManager()
            llm_kwargs: dict = {}
            if temperature is not None:
                llm_kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                llm_kwargs["max_tokens"] = int(max_tokens)
            if model_override:
                llm_kwargs["model"] = model_override

            response = await llm_manager.generate(
                prompt=user_prompt,
                model_type=model_type,
                system_prompt=str(agent.system_prompt),
                **llm_kwargs,
            )

            # Step 7 - store result
            content: str = response.get("content", "")
            usage: dict = response.get("usage", {})
            cost = response.get("cost")

            run.result = content
            run.tokens_used = usage.get("total_tokens")
            run.estimated_cost_usd = float(cost) if cost is not None else None
            run.model_used = response.get("model")

            # Step 8 - try to parse structured JSON from result
            try:
                # Strip common markdown code fences before parsing
                stripped = content.strip()
                if stripped.startswith("```"):
                    # Remove opening fence (```json or ```)
                    first_newline = stripped.find("\n")
                    if first_newline != -1:
                        stripped = stripped[first_newline + 1:]
                    # Remove closing fence
                    if stripped.rstrip().endswith("```"):
                        stripped = stripped.rstrip()[:-3].rstrip()
                run.result_structured = json.loads(stripped)
            except Exception:
                # Silent failure - structured parsing is best-effort
                pass

            # Step 9 - mark COMPLETED
            run.status = InvestigationRunStatus.COMPLETED
            await self.db.commit()
            await self.db.refresh(run)

            logger.info(
                "Investigation completed: run=%s tokens=%s cost=$%.6f",
                run.id,
                run.tokens_used,
                run.estimated_cost_usd or 0.0,
            )

        except Exception as exc:
            logger.exception(
                "Investigation failed: agent=%s idea=%s error=%s",
                agent_id,
                idea_id,
                exc,
            )
            run.status = InvestigationRunStatus.FAILED
            run.error_message = str(exc)
            await self.db.commit()

        return run

    async def get_runs_for_idea(self, idea_id: UUID) -> list[InvestigationRun]:
        """Return all investigation runs for an idea, ordered newest first."""
        stmt = (
            select(InvestigationRun)
            .where(InvestigationRun.idea_id == idea_id)
            .order_by(InvestigationRun.created_at.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_run(self, run_id: UUID) -> InvestigationRun:
        """Return a single investigation run by ID."""
        run = await self.db.get(InvestigationRun, run_id)
        if run is None:
            raise NotFoundException("InvestigationRun", str(run_id))
        return run
