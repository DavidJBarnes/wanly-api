"""title_tags.description; Me moves to the Person group (wanly-api#486)

Group 1 is Person tags and group 2 is Action tags. An Action tag's description is the short
"what happens" prompt Enhance prompt sends to the Sulphur-2 enhancer. Me is a person, so it
moves to group 1 -- or, if a group-1 Me already exists, the group-2 duplicate is dropped
(image tags are stored by name, so no image loses its tag either way).

Revision ID: 124
Revises: 123
"""
import sqlalchemy as sa
from alembic import op

revision = "124"
down_revision = "123"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("title_tags", sa.Column("description", sa.Text(), nullable=True))
    op.execute("""DELETE FROM title_tags WHERE lower(name) = 'me' AND "group" = 2
                  AND EXISTS (SELECT 1 FROM title_tags WHERE lower(name) = 'me' AND "group" = 1)""")
    op.execute("""UPDATE title_tags SET "group" = 1 WHERE lower(name) = 'me' AND "group" = 2""")


def downgrade() -> None:
    op.execute("""UPDATE title_tags SET "group" = 2 WHERE lower(name) = 'me' AND "group" = 1""")
    op.drop_column("title_tags", "description")
