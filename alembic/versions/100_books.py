"""Books: group poses into named collections

Revision ID: 100
Revises: 099
Create Date: 2026-09-14

A pose stops being a row in one flat list and gets a shelf. `ltx_books` holds names and
descriptions; `ltx_recipes.book_id` is mandatory and RESTRICTs deletion, so a populated book
cannot be deleted by accident.

WHY MANDATORY, GIVEN 072 REMOVED A PARENT FK
    Migration 072 dropped the character FK because a mandatory parent locked new LoRAs out:
    a character with no rows had no poses to add. The shape is different here and so is the
    failure it would cause. A book is not a filter over poses, it is where a pose lives, and
    the route defaults a new pose into the default book rather than refusing it. A pose can
    still never be orphaned (NOT NULL) and a new pose can still always be created (default).
    The alternative -- nullable book_id -- would let "which book is this in" be unanswerable,
    which is the flat list again with extra steps.

BACKFILL: ONE BOOK PER CHECKPOINT FAMILY, NOT ONE DEFAULT BOOK
    The poses already name their base model in `checkpoint`, and that is the grouping that
    exists in the data. Nine name 10Eros, six name sulphur. David's rule: "Any Pose currently
    using base model 10eros can go to a new book entitled 10eros, same for sulphur."

    A NULL checkpoint means "use the stack's value", which is 10Eros_v1.5_bf16 (app/ltx_stack),
    so those poses land in the 10eros book too. The mapping is inlined and frozen here rather
    than imported: a later edit to the stack default must not rewrite what this migration did.

    Names are matched case-insensitively on a substring, then fall back to the checkpoint
    filename with any extension stripped. An unknown checkpoint therefore gets its own book
    under its own name rather than being silently filed somewhere wrong.
"""
import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "100"
down_revision = "099"
branch_labels = None
depends_on = None

# Checkpoint substring (lowercased) -> the book it belongs to. Order matters only in that
# every listed key is distinct enough not to overlap; first match wins.
_FAMILIES = [
    ("10eros", "10eros"),
    ("sulphur", "sulphur"),
]

# What a NULL checkpoint means, matching LTX_STACK["checkpoint"]. Frozen here on purpose.
_STACK_DEFAULT_CHECKPOINT = "10Eros_v1.5_bf16"
_DEFAULT_BOOK = "10eros"


def _book_for(checkpoint: str | None) -> tuple[str, str | None]:
    """Return (book name, the checkpoint that put it there or None for the stack default)."""
    raw = (checkpoint or "").strip()
    if not raw:
        return _DEFAULT_BOOK, _STACK_DEFAULT_CHECKPOINT
    low = raw.lower()
    for needle, book in _FAMILIES:
        if needle in low:
            return book, raw
    name = raw[: -len(".safetensors")] if raw.endswith(".safetensors") else raw
    return name, raw


def upgrade() -> None:
    op.create_table(
        "ltx_books",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("name", name="uq_ltx_books_name"),
    )

    op.add_column("ltx_recipes", sa.Column("book_id", UUID(as_uuid=True), nullable=True))

    conn = op.get_bind()
    existing = conn.execute(sa.text("SELECT DISTINCT checkpoint FROM ltx_recipes")).fetchall()

    # book name -> id, seeded from whatever checkpoints are actually in the table.
    book_ids: dict[str, uuid.UUID] = {}
    description = {
        _DEFAULT_BOOK: f"Poses on {_STACK_DEFAULT_CHECKPOINT}.",
    }
    for (checkpoint,) in existing:
        name, source = _book_for(checkpoint)
        if name in book_ids:
            continue
        book_ids[name] = uuid.uuid4()
        if name not in description:
            description[name] = f"Poses on {source}."

    # A database with no poses at all still needs the default book, or the first create has
    # nowhere to land.
    if _DEFAULT_BOOK not in book_ids:
        book_ids[_DEFAULT_BOOK] = uuid.uuid4()
        description.setdefault(_DEFAULT_BOOK, f"Poses on {_STACK_DEFAULT_CHECKPOINT}.")

    for name, bid in book_ids.items():
        conn.execute(
            sa.text("INSERT INTO ltx_books (id, name, description) VALUES (:id, :name, :d)"),
            {"id": bid, "name": name, "d": description.get(name)},
        )

    # Point each pose at the book derived from its own checkpoint. Done per book so the
    # NULL-checkpoint poses and the named ones are handled by the same rule.
    for name, bid in book_ids.items():
        if name == _DEFAULT_BOOK:
            # The default book takes both the NULLs and anything naming 10Eros.
            conn.execute(
                sa.text(
                    "UPDATE ltx_recipes SET book_id = :bid "
                    "WHERE checkpoint IS NULL OR lower(checkpoint) LIKE :like"
                ),
                {"bid": bid, "like": "%10eros%"},
            )
        else:
            conn.execute(
                sa.text(
                    "UPDATE ltx_recipes SET book_id = :bid WHERE lower(checkpoint) LIKE :like"
                ),
                {"bid": bid, "like": f"%{name}%"},
            )

    # Nothing may be left unfiled. If this trips, the family rule above missed a checkpoint.
    orphans = conn.execute(
        sa.text("SELECT count(*) FROM ltx_recipes WHERE book_id IS NULL")
    ).scalar()
    if orphans:
        raise RuntimeError(f"{orphans} poses left without a book; refine _book_for")

    op.alter_column("ltx_recipes", "book_id", nullable=False)
    op.create_foreign_key(
        "fk_ltx_recipes_book_id", "ltx_recipes", "ltx_books",
        ["book_id"], ["id"], ondelete="RESTRICT",
    )
    op.create_index("ix_ltx_recipes_book_id", "ltx_recipes", ["book_id"])

    op.drop_constraint("uq_ltx_recipe_name", "ltx_recipes", type_="unique")
    op.create_unique_constraint(
        "uq_ltx_recipe_book_name", "ltx_recipes", ["book_id", "name"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_ltx_recipe_book_name", "ltx_recipes", type_="unique")
    op.create_unique_constraint("uq_ltx_recipe_name", "ltx_recipes", ["name"])

    op.drop_index("ix_ltx_recipes_book_id", table_name="ltx_recipes")
    op.drop_constraint("fk_ltx_recipes_book_id", "ltx_recipes", type_="foreignkey")
    op.drop_column("ltx_recipes", "book_id")
    op.drop_table("ltx_books")
