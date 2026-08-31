"""A recipe is a POSE, not a pose-per-character

Revision ID: 072
Revises: 071
Create Date: 2026-08-31

071 stored 24 recipes: 8 poses x 3 characters. That was wrong, and it locked new LoRAs out —
a character with no rows in the table had no recipes at all, so adding one meant duplicating
eight prompts rather than naming a LoRA and a trigger word.

**Recipes were never meant to be LoRA-specific.** Verified against the seeded data before
changing it: strip the leading trigger token and all 8 poses are character-agnostic. The only
"differences" between the three copies of Cowgirl POV and Doggystyle Side were a stray
trailing space — three copies of one prompt that had already begun to drift, which is the
argument for this change rather than an objection to it.

So: 8 poses, each carrying a <TRIGGER> placeholder, and a character supplies the LoRA and
the trigger word that fills it. Adding a character costs a row, and every pose works for it
immediately.

<TRIGGER> shares syntax with the wildcard resolver (`<([^<>]+)>` in
app/routes/segments.py::_resolve_wildcards). An unmatched name is left alone, so this is safe
today — but a Wildcard named TRIGGER would make the resolver swap in a RANDOM option, and the
render would quietly name the wrong character. The trigger is therefore substituted BEFORE
wildcard resolution, and the name is reserved so that wildcard cannot be created.

`validated` moves meaning slightly and deliberately. It marks the POSE as proven: the prompt
produces what it claims. Whether a given CHARACTER renders well is a property of its LoRA, and
that is what segment ratings and observations already record. p@y's rows were marked
unvalidated in the sheet because p@y was new, not because the poses were in doubt.
"""
import sqlalchemy as sa
from alembic import op


# Frozen: a migration records what happened, so its data must not be importable and editable.
_POSES = [{'name': 'Blowjob Other', 'prompt_template': '<TRIGGER>, a woman kneeling beside a nude man seen from the side, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and bobs her head all the way down and all the way back up, glancing toward the camera. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Blowjob POV', 'prompt_template': '<TRIGGER>, a woman kneeling in front of a nude man, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and, maintaining eye contact with the viewer, bobs her head all the way down and all the way back up. Her eyes stay locked on the viewer. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Cowgirl Backside', 'prompt_template': '<TRIGGER>, a woman having reverse cowgirl sex seen from behind, she is on top of a man facing away from the camera, his penis is clearly seen gliding all the way in and all the way back out of her as she rides him, her buttocks and back moving with each stroke. Her face stays in focus the entire time. She looks back over her shoulder toward the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Cowgirl POV', 'prompt_template': '<TRIGGER>, a woman having cowgirl sex, she is on top of a man, his penis is clearly seen smoothly gliding all the way in and all the way back out of her vagina, her breasts bounce and jiggle as she rides and grinds on him. Her facial expressions show intense pleasure and she maintains eye contact with the viewer. Her face stays in focus the entire time.  Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Doggystyle Front Facing', 'prompt_template': '<TRIGGER>, a woman on her knees bent over a bed is facing the camera. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. She maintains steady eye contact with the camera with natural expressions that align with the encounter. Her face stays in focus the entire time. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Doggystyle Side', 'prompt_template': '<TRIGGER>, a woman on her knees bent over a bed seen from the side. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. Her face is turned toward the camera with natural expressions that align with the encounter. Her face stays in focus the entire time.  Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Missionary POV', 'prompt_template': '<TRIGGER>, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}, {'name': 'Missionary Side', 'prompt_template': '<TRIGGER>, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.'}]

_TRIGGERS = {'k3lly2026': 'k3lly2026', 'k3llydw': 'k3llydw', 'p@y': 'p@y'}

revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # A character supplies the trigger word that fills <TRIGGER>. Seeded from the leading
    # token of its own prompts, which is where it lived before.
    op.add_column("ltx_characters", sa.Column("trigger", sa.String(length=64), nullable=True))
    for name, trig in _TRIGGERS.items():
        conn.execute(sa.text("UPDATE ltx_characters SET trigger = :t WHERE name = :n"),
                     {"t": trig, "n": name})
    # Backfill anything unseeded with its own name, then make it required: a character with
    # no trigger renders a prompt containing a literal placeholder, which is worse than
    # failing loudly at creation.
    conn.execute(sa.text("UPDATE ltx_characters SET trigger = name WHERE trigger IS NULL"))
    op.alter_column("ltx_characters", "trigger", nullable=False)

    # Poses are no longer per character.
    op.drop_index("ix_ltx_recipes_character_id", table_name="ltx_recipes")
    op.drop_constraint("uq_ltx_recipe_character_name", "ltx_recipes", type_="unique")
    op.drop_column("ltx_recipes", "character_id")
    op.alter_column("ltx_recipes", "prompt", new_column_name="prompt_template")

    # 24 rows collapse to 8. Replaced rather than deduped in SQL: the templates below are the
    # authoritative text, and picking a survivor row would preserve whichever copy happened to
    # carry the stray whitespace.
    #
    # DELETE before the unique constraint, not after. The 24 rows are 8 names x 3 characters,
    # so adding UNIQUE(name) while they are still present fails outright — which it did.
    conn.execute(sa.text("DELETE FROM ltx_recipes"))
    import uuid as _uuid
    for p in _POSES:
        conn.execute(
            sa.text("INSERT INTO ltx_recipes (id, name, prompt_template, validated) "
                    "VALUES (:id, :name, :prompt, true)"),
            {"id": _uuid.uuid4(), "name": p["name"], "prompt": p["prompt_template"]},
        )

    # Only once the duplicates are gone.
    op.create_unique_constraint("uq_ltx_recipe_name", "ltx_recipes", ["name"])


def downgrade() -> None:
    # Not reversible in any useful sense: 8 poses cannot be expanded back into 24 rows without
    # knowing which characters existed at the time. Restores the shape, not the data.
    op.drop_constraint("uq_ltx_recipe_name", "ltx_recipes", type_="unique")
    op.alter_column("ltx_recipes", "prompt_template", new_column_name="prompt")
    op.add_column("ltx_recipes", sa.Column("character_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_ltx_recipes_character_id", "ltx_recipes", ["character_id"])
    op.drop_column("ltx_characters", "trigger")
