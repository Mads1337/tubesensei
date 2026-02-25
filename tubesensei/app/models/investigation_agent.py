from sqlalchemy import Column, String, Text, Boolean
from sqlalchemy.dialects.postgresql import JSONB

from app.models.base import BaseModel


class InvestigationAgent(BaseModel):
    __tablename__ = "investigation_agents"

    name = Column(
        String(200),
        nullable=False
    )

    description = Column(
        Text,
        nullable=True
    )

    system_prompt = Column(
        Text,
        nullable=False
    )

    user_prompt_template = Column(
        Text,
        nullable=False
    )

    config = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}"
    )

    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True
    )

    def __repr__(self) -> str:
        return f"<InvestigationAgent(id={self.id}, name={self.name}, is_active={self.is_active})>"
