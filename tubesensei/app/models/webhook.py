"""Webhook subscription model for external notifications."""
from sqlalchemy import Column, String, Boolean, DateTime, Integer
from sqlalchemy.dialects.postgresql import ARRAY

from app.models.base import BaseModel


class Webhook(BaseModel):
    __tablename__ = "webhooks"

    name = Column(
        String(255),
        nullable=False,
    )

    url = Column(
        String(500),
        nullable=False,
    )

    secret = Column(
        String(255),
        nullable=True,
    )

    events = Column(
        ARRAY(String()),
        nullable=False,
        default=list,
        server_default="{}",
    )

    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        index=True,
    )

    last_triggered_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    failure_count = Column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    def __repr__(self) -> str:
        return f"<Webhook(id={self.id}, name={self.name}, url={self.url}, is_active={self.is_active})>"
