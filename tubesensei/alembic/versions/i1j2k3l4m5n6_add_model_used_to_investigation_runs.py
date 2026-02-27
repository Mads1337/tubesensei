"""Add model_used to investigation_runs

Revision ID: i1j2k3l4m5n6
Revises: add_webhooks_table
Create Date: 2026-02-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'i1j2k3l4m5n6'
down_revision: Union[str, Sequence[str], None] = 'add_webhooks_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'investigation_runs',
        sa.Column('model_used', sa.String(100), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('investigation_runs', 'model_used')
