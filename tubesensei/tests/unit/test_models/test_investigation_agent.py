"""
Unit tests for the InvestigationAgent model.

Tests creation and field validation.
"""
import pytest

from app.models.investigation_agent import InvestigationAgent


class TestInvestigationAgentCreation:
    """Tests for InvestigationAgent model creation."""

    @pytest.mark.asyncio
    async def test_basic_creation(self, db_session):
        """Test creating an InvestigationAgent with required fields."""
        agent = InvestigationAgent(
            name="Market Analyzer",
            description="Analyzes market opportunity for an idea",
            system_prompt="You are a market research expert.",
            user_prompt_template="Analyze this idea: {idea_title}\n\n{idea_description}",
            config={"model_type": "balanced"},
            is_active=True,
        )
        db_session.add(agent)
        await db_session.commit()

        assert agent.id is not None
        assert agent.name == "Market Analyzer"
        assert agent.description == "Analyzes market opportunity for an idea"
        assert agent.is_active is True
        assert agent.config == {"model_type": "balanced"}

    @pytest.mark.asyncio
    async def test_creation_without_description(self, db_session):
        """Test creating an InvestigationAgent without optional description."""
        agent = InvestigationAgent(
            name="Minimal Agent",
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze: {idea_title}",
            config={},
            is_active=True,
        )
        db_session.add(agent)
        await db_session.commit()

        assert agent.id is not None
        assert agent.description is None

    @pytest.mark.asyncio
    async def test_is_active_default_true(self, db_session):
        """Test that is_active defaults to True."""
        agent = InvestigationAgent(
            name="Default Active Agent",
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze: {idea_title}",
            config={},
        )
        db_session.add(agent)
        await db_session.commit()

        assert agent.is_active is True

    @pytest.mark.asyncio
    async def test_can_be_set_inactive(self, db_session):
        """Test that is_active can be set to False."""
        agent = InvestigationAgent(
            name="Inactive Agent",
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze: {idea_title}",
            config={},
            is_active=False,
        )
        db_session.add(agent)
        await db_session.commit()

        assert agent.is_active is False

    @pytest.mark.asyncio
    async def test_config_stores_arbitrary_data(self, db_session):
        """Test that config JSONB field can store arbitrary data."""
        config_data = {
            "model_type": "quality",
            "temperature": 0.7,
            "max_tokens": 2000,
            "custom_param": "value",
        }
        agent = InvestigationAgent(
            name="Configured Agent",
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze: {idea_title}",
            config=config_data,
            is_active=True,
        )
        db_session.add(agent)
        await db_session.commit()

        assert agent.config["model_type"] == "quality"
        assert agent.config["temperature"] == 0.7
        assert agent.config["max_tokens"] == 2000

    @pytest.mark.asyncio
    async def test_has_created_at_and_updated_at(self, db_session):
        """Test that timestamps are set automatically."""
        agent = InvestigationAgent(
            name="Timestamps Agent",
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze: {idea_title}",
            config={},
            is_active=True,
        )
        db_session.add(agent)
        await db_session.commit()

        assert agent.created_at is not None
        assert agent.updated_at is not None

    def test_repr(self):
        """Test string representation."""
        agent = InvestigationAgent(
            name="Test Agent",
            system_prompt="prompt",
            user_prompt_template="template",
            is_active=True,
        )
        repr_str = repr(agent)
        assert "InvestigationAgent" in repr_str
        assert "Test Agent" in repr_str
