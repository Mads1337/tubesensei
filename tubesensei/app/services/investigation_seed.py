import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.investigation_agent import InvestigationAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template helpers
# ---------------------------------------------------------------------------

_IDEA_CONTEXT_BLOCK = """\
Analyze the following idea:

**Title:** {idea_title}
**Description:** {idea_description}
**Category:** {idea_category}
**Tags:** {idea_tags}
**Technologies:** {idea_technologies}
**Confidence Score:** {idea_confidence_score}
**Complexity Score:** {idea_complexity_score}
**Market Size Estimate:** {idea_market_size}
**Target Audience:** {idea_target_audience}
**Competitive Advantage:** {idea_competitive_advantage}
**Potential Challenges:** {idea_challenges}
**Monetization Strategies:** {idea_monetization}"""

_FORMAT_FOOTER = (
    "\n\nFormat your response as a structured analysis with clear sections "
    "and actionable insights."
)


def _build_template(analysis_type: str, bullet_points: str) -> str:
    return (
        _IDEA_CONTEXT_BLOCK
        + f"\n\nPlease provide a detailed {analysis_type} covering:\n"
        + bullet_points
        + _FORMAT_FOOTER
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_AGENTS: list[dict] = [
    {
        "name": "Financial Analysis",
        "description": (
            "Analyzes the financial viability, revenue potential, and cost "
            "structure of an idea"
        ),
        "system_prompt": (
            "You are an expert financial analyst specializing in technology "
            "startups and digital products. Analyze business ideas with focus "
            "on revenue models, unit economics, funding requirements, "
            "break-even analysis, and ROI projections. Be specific with "
            "numbers where possible, and always identify key financial risks."
        ),
        "user_prompt_template": _build_template(
            "financial analysis",
            "- Revenue model assessment\n"
            "- Cost structure analysis\n"
            "- Break-even timeline\n"
            "- Funding requirements\n"
            "- ROI projections\n"
            "- Financial risks and mitigation strategies",
        ),
        "config": {"model_type": "quality", "temperature": 0.3, "max_tokens": 2000},
    },
    {
        "name": "Feasibility Study",
        "description": (
            "Evaluates technical and operational feasibility including "
            "resource requirements and timeline"
        ),
        "system_prompt": (
            "You are a seasoned technical project manager and systems "
            "architect. Evaluate ideas for technical feasibility, resource "
            "requirements, implementation timeline, and potential technical "
            "roadblocks. Consider both MVP and full-scale implementation "
            "paths. Be practical and realistic in your assessments."
        ),
        "user_prompt_template": _build_template(
            "feasibility study",
            "- Technical feasibility assessment\n"
            "- Resource requirements (team, infrastructure, tools)\n"
            "- Implementation timeline (MVP + full launch)\n"
            "- Key dependencies and risks\n"
            "- Go/no-go recommendation with reasoning",
        ),
        "config": {"model_type": "quality", "temperature": 0.3, "max_tokens": 2000},
    },
    {
        "name": "Market Research",
        "description": (
            "Conducts market analysis including TAM/SAM/SOM, competitor "
            "landscape, and market trends"
        ),
        "system_prompt": (
            "You are a market research analyst with deep expertise in "
            "technology markets. Analyze ideas by examining market size "
            "(TAM/SAM/SOM), competitor landscape, market trends, customer "
            "segments, and go-to-market strategies. Use frameworks like "
            "Porter's Five Forces where applicable. Provide actionable "
            "market intelligence."
        ),
        "user_prompt_template": _build_template(
            "market research",
            "- Market size analysis (TAM/SAM/SOM)\n"
            "- Target customer segments and personas\n"
            "- Competitor mapping\n"
            "- Market trends and timing\n"
            "- Go-to-market strategy recommendations\n"
            "- Market risks and opportunities",
        ),
        "config": {"model_type": "quality", "temperature": 0.4, "max_tokens": 2000},
    },
    {
        "name": "Technical Complexity",
        "description": (
            "Assesses technical architecture, stack requirements, scalability, "
            "and engineering challenges"
        ),
        "system_prompt": (
            "You are a principal software engineer and solution architect. "
            "Evaluate ideas from a technical perspective, assessing "
            "architecture patterns, technology stack choices, scalability "
            "concerns, security implications, and engineering team "
            "requirements. Provide concrete technical recommendations and "
            "identify potential pitfalls."
        ),
        "user_prompt_template": _build_template(
            "technical complexity assessment",
            "- Recommended architecture and tech stack\n"
            "- Scalability assessment\n"
            "- Security considerations\n"
            "- Integration requirements\n"
            "- Engineering team size and skill requirements\n"
            "- Technical debt risks\n"
            "- Infrastructure cost estimates",
        ),
        "config": {"model_type": "quality", "temperature": 0.2, "max_tokens": 2000},
    },
    {
        "name": "Competitive Analysis",
        "description": (
            "Maps the competitive landscape, identifies differentiators, "
            "and evaluates market positioning"
        ),
        "system_prompt": (
            "You are a competitive intelligence analyst specializing in "
            "technology markets. Analyze ideas by mapping direct and indirect "
            "competitors, identifying unique differentiators, evaluating "
            "barriers to entry, and assessing sustainable competitive "
            "advantages. Focus on actionable strategic positioning "
            "recommendations."
        ),
        "user_prompt_template": _build_template(
            "competitive analysis",
            "- Direct competitor analysis\n"
            "- Indirect competitor and substitute analysis\n"
            "- Competitive advantages and moats\n"
            "- Barriers to entry\n"
            "- SWOT analysis\n"
            "- Strategic positioning recommendations",
        ),
        "config": {"model_type": "quality", "temperature": 0.3, "max_tokens": 2000},
    },
]


# ---------------------------------------------------------------------------
# Seed function
# ---------------------------------------------------------------------------


async def seed_investigation_agents(db: AsyncSession) -> list[InvestigationAgent]:
    """Seed default investigation agent templates. Creates new agents or updates existing ones if config changed."""
    created: list[InvestigationAgent] = []
    updated: list[InvestigationAgent] = []

    for agent_data in _AGENTS:
        name = agent_data["name"]

        existing = await db.scalar(
            select(InvestigationAgent).where(InvestigationAgent.name == name)
        )
        if existing:
            # Update config if it has changed
            if existing.config != agent_data["config"]:
                existing.config = agent_data["config"]
                updated.append(existing)
                logger.info("Updated config for investigation agent '%s'.", name)
            else:
                logger.info("Investigation agent '%s' already up to date — skipping.", name)
            continue

        agent = InvestigationAgent(
            name=name,
            description=agent_data["description"],
            system_prompt=agent_data["system_prompt"],
            user_prompt_template=agent_data["user_prompt_template"],
            config=agent_data["config"],
            is_active=True,
        )
        db.add(agent)
        created.append(agent)
        logger.info("Queued investigation agent '%s' for insertion.", name)

    if created or updated:
        await db.commit()
        for agent in created:
            await db.refresh(agent)
        logger.info(
            "Seeded %d new, updated %d investigation agent(s).",
            len(created),
            len(updated),
        )
    else:
        logger.info("No investigation agents to seed or update.")

    return created
