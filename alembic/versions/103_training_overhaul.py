"""Training overhaul: dataset ownership, per-image captions and scores, pair characters (#352)

Revision ID: 103
Revises: 102
Create Date: 2026-09-27

WHAT WAS WRONG, and what each column is for:

  datasets.kind / character / reg_class
      A dataset was a bag of images with a name. Nothing said WHOSE face it held, so the
      "David" LoRA trained on 14 images of somebody else and nothing could have noticed.
      `kind` is what the set is for (character | composition | regularization), `character`
      is its owner (for a composition set, the PAIR's name), `reg_class` is the class word a
      regularization pool stands in for (woman | man). NULL kind is "unassigned", which the
      console flags and the training route refuses.

  datasets.captions  {uri: body}
      Every image trained under one caption, "<trigger>, <gender>", so framing, clothing,
      lighting and background were all absorbed into the trigger. The BODY is stored per
      image WITHOUT the trigger; the prefix is added when a run is created, so the same set
      can never end up trained under the wrong trigger. Keyed by URI, not index: the list is
      reordered by removal, and an index would silently come to caption another photograph.

  datasets.scores  {uri: cos}
      POST /datasets/{id}/score computed likeness to the anchor and threw it away. The
      training route needs it to refuse a set with somebody else's face in it, so it is kept.

  ltx_characters.trigger -> 255
      A pair's trigger is the joined phrase "d@vid, man and k3lly2026, woman", which does not
      fit in 64 once real names are involved.

  ltx_characters.kind / members / base_checkpoint
      A joint run used to publish over group 0's SOLO row, so training "DavidKelly" replaced
      what "David" rendered with. A pair is now its own row (kind='pair', members=[names]).
      base_checkpoint records what the LoRA was trained against -- a LoRA trained on dev and
      rendered on 10Eros is exactly the mismatch this overhaul exists to end.

BACKFILL
  Datasets whose name matches a character's (case-insensitively) become kind='character'
  owned by that character. Everything else stays unassigned: guessing an owner is how the
  wrong face got into a LoRA in the first place.

  A character becomes a pair only when its last completed training job names TWO OTHER
  registered characters as its identity groups. Every legacy joint run published under a
  member's own name, so in practice that is rare and most rows stay solo -- a row whose
  current LoRA is joint but whose name is also a member cannot be both, and is left for a
  human to split rather than being half-converted here.
"""
import json

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "103"
down_revision = "102"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("kind", sa.String(20), nullable=True))
    op.add_column("datasets", sa.Column("character", sa.String(64), nullable=True))
    op.add_column("datasets", sa.Column("reg_class", sa.String(16), nullable=True))
    op.add_column("datasets", sa.Column("captions", postgresql.JSONB(), nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))
    op.add_column("datasets", sa.Column("scores", postgresql.JSONB(), nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))

    op.alter_column("ltx_characters", "trigger", type_=sa.String(255),
                    existing_type=sa.String(64), existing_nullable=False)
    op.add_column("ltx_characters", sa.Column("kind", sa.String(16), nullable=False,
                                              server_default="solo"))
    op.add_column("ltx_characters", sa.Column("members", postgresql.JSONB(), nullable=True))
    op.add_column("ltx_characters", sa.Column("base_checkpoint", sa.Text(), nullable=True))

    op.execute("""
        UPDATE datasets AS d
           SET kind = 'character', character = c.name
          FROM ltx_characters AS c
         WHERE lower(d.name) = lower(c.name)
    """)

    _backfill_pairs()


def _backfill_pairs() -> None:
    bind = op.get_bind()
    chars = bind.execute(sa.text("SELECT id, name, trigger FROM ltx_characters")).all()
    by_name = {c.name: c for c in chars}
    by_trigger: dict[str, str] = {}
    for c in chars:
        # A trigger that is itself a joined phrase is not one person's, and must not be
        # mistaken for the trigger of whoever it starts with.
        if c.trigger and " and " not in c.trigger:
            by_trigger.setdefault(c.trigger, c.name)

    for c in chars:
        job = bind.execute(sa.text("""
            SELECT character, trigger, identities FROM training_jobs
             WHERE character = :name AND status = 'completed'
             ORDER BY completed_at DESC NULLS LAST, created_at DESC
             LIMIT 1
        """), {"name": c.name}).first()
        if job is None:
            continue
        groups = job.identities
        if isinstance(groups, str):
            groups = json.loads(groups)
        if not groups:
            continue
        members: list[str] = []

        def _add(name: str | None) -> None:
            if name and name in by_name and name not in members:
                members.append(name)

        # Group 0's person is whoever owns its trigger -- the job's `character` is the row
        # it published to, which is exactly what is in question.
        _add(by_trigger.get(job.trigger))
        for g in groups:
            if not isinstance(g, dict) or not g.get("trigger"):
                continue  # a composition group names nobody new
            _add(g.get("character") if g.get("character") in by_name
                 else by_trigger.get(g.get("trigger")))
        if len(members) == 2 and c.name not in members:
            bind.execute(sa.text(
                "UPDATE ltx_characters SET kind = 'pair', members = CAST(:m AS jsonb) "
                "WHERE id = :id"), {"m": json.dumps(members), "id": c.id})


def downgrade() -> None:
    op.drop_column("ltx_characters", "base_checkpoint")
    op.drop_column("ltx_characters", "members")
    op.drop_column("ltx_characters", "kind")
    # A pair phrase longer than 64 cannot survive the narrower column; truncating it is the
    # lesser evil against a downgrade that cannot run at all.
    op.alter_column("ltx_characters", "trigger", type_=sa.String(64),
                    existing_type=sa.String(255), existing_nullable=False,
                    postgresql_using="left(trigger, 64)")
    op.drop_column("datasets", "scores")
    op.drop_column("datasets", "captions")
    op.drop_column("datasets", "reg_class")
    op.drop_column("datasets", "character")
    op.drop_column("datasets", "kind")
