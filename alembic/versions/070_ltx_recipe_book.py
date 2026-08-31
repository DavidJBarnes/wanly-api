"""The LTX recipe book lives in the API

Revision ID: 070
Revises: 069
Create Date: 2026-08-31

An LTX render is driven by a validated (character, pose) configuration authored in a
hand-maintained .ods sheet. Until now that sheet was parsed into a recipes.json which existed
in TWO copies — the repo's and the engine's — kept in step by a Dockerfile COPY. Switching the
engine to a bind mount removed the rebuild, and with it the copy. Nothing replaced it. A
16-render batch then ran against stale prompts, and a finding was drawn from those renders,
written up as established, and agreed to before being retracted.

The check that missed it called GET /recipes and confirmed it returned 8 recipes with the
right names. Both were equally true of the stale file. It confirmed facts that had not changed
instead of content that had.

So: one authority. The API holds the book, the console reads it from here, and the engine
receives the RESOLVED recipe inside the claim rather than looking one up. An engine that
cannot look a recipe up cannot look up a stale one — the drift becomes structurally
impossible rather than something a check has to catch.

A single row, because there is one book. `source_sha256` is over the uploaded .ods bytes and
`book_sha256` over the parsed content — the second is the one that matters, since a
re-saved sheet with no edits changes the file bytes but not the book.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "070"
down_revision = "069"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ltx_recipe_book",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("book", JSONB(), nullable=False),
        sa.Column("book_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        # One book. The constraint is the documentation: this is a singleton, not a history.
        sa.CheckConstraint("id = 1", name="ck_ltx_recipe_book_singleton"),
    )


def downgrade() -> None:
    op.drop_table("ltx_recipe_book")
