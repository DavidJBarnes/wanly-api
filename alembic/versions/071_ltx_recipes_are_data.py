"""LTX recipes become data, not an imported spreadsheet

Revision ID: 071
Revises: 070
Create Date: 2026-08-31

Replaces the sheet-backed recipe book from 070.

The .ods was a **test harness that became load-bearing**. The POC built guards around it —
"never regenerate the sheet", parse defensively, hash the result — which is what carrying a
harness forward looks like rather than replacing it. Recipes are data: rows, created and
edited in the console like LoRAs and video presets.

**The schema encodes what was actually measured.** Across all 24 recipes, exactly two fields
varied: `char_lora` (3 distinct — one per character) and `prompt` (24). Checkpoint, per-stage
strengths, content LoRA, distill, guidance, steps, frames, resolution and the negative prompt
each had ONE value. So a recipe is `(character, prompt)` and the rest is one global stack,
which lives once in app_settings rather than being copied 24 times.

Storing it 24 times is how a "global" value silently stops being global: one row gets edited,
nothing complains, and two recipes that should be identical are not.

Prompts are seeded RESOLVED. The sheet stored keys (`P.MISS_POV`) against a definitions tab;
that indirection existed to avoid retyping text in a spreadsheet and buys nothing in a
database, where the console edits the text directly.
"""
import sqlalchemy as sa
from alembic import op


# The 24 recipes as they stood in the POC sheet, INLINED rather than imported.
#
# A migration is a record of what happened, so its data has to be frozen. Importing this from
# a module would mean a later edit to that module silently changes what this migration did,
# which is the opposite of the point.
_CHARACTERS = [{'name': 'k3lly2026', 'char_lora': 'k3lly2026_v2', 'strength_stage_1': 0.8, 'strength_stage_2': 1.5}, {'name': 'k3llydw', 'char_lora': 'k3llydw_v2', 'strength_stage_1': 0.8, 'strength_stage_2': 1.5}, {'name': 'p@y', 'char_lora': 'pay_v2_e05', 'strength_stage_1': 0.8, 'strength_stage_2': 1.5}]

_RECIPES = [{'character': 'k3lly2026', 'name': 'Missionary POV', 'prompt': 'k3lly2026, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Missionary Side', 'prompt': 'k3lly2026, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Doggystyle Front Facing', 'prompt': 'k3lly2026, a woman on her knees bent over a bed is facing the camera. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. She maintains steady eye contact with the camera with natural expressions that align with the encounter. Her face stays in focus the entire time. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Doggystyle Side', 'prompt': 'k3lly2026, a woman on her knees bent over a bed seen from the side. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. Her face is turned toward the camera with natural expressions that align with the encounter. Her face stays in focus the entire time.  Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Cowgirl POV', 'prompt': 'k3lly2026, a woman having cowgirl sex, she is on top of a man, his penis is clearly seen smoothly gliding all the way in and all the way back out of her vagina, her breasts bounce and jiggle as she rides and grinds on him. Her facial expressions show intense pleasure and she maintains eye contact with the viewer. Her face stays in focus the entire time.  Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Cowgirl Backside', 'prompt': 'k3lly2026, a woman having reverse cowgirl sex seen from behind, she is on top of a man facing away from the camera, his penis is clearly seen gliding all the way in and all the way back out of her as she rides him, her buttocks and back moving with each stroke. Her face stays in focus the entire time. She looks back over her shoulder toward the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Blowjob POV', 'prompt': 'k3lly2026, a woman kneeling in front of a nude man, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and, maintaining eye contact with the viewer, bobs her head all the way down and all the way back up. Her eyes stay locked on the viewer. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3lly2026', 'name': 'Blowjob Other', 'prompt': 'k3lly2026, a woman kneeling beside a nude man seen from the side, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and bobs her head all the way down and all the way back up, glancing toward the camera. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Missionary POV', 'prompt': 'k3llydw, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Missionary Side', 'prompt': 'k3llydw, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Doggystyle Front Facing', 'prompt': 'k3llydw, a woman on her knees bent over a bed is facing the camera. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. She maintains steady eye contact with the camera with natural expressions that align with the encounter. Her face stays in focus the entire time. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Doggystyle Side', 'prompt': 'k3llydw, a woman on her knees bent over a bed seen from the side. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. Her face is turned toward the camera with natural expressions that align with the encounter. Her face stays in focus the entire time. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Cowgirl POV', 'prompt': 'k3llydw, a woman having cowgirl sex, she is on top of a man, his penis is clearly seen smoothly gliding all the way in and all the way back out of her vagina, her breasts bounce and jiggle as she rides and grinds on him. Her facial expressions show intense pleasure and she maintains eye contact with the viewer. Her face stays in focus the entire time. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Cowgirl Backside', 'prompt': 'k3llydw, a woman having reverse cowgirl sex seen from behind, she is on top of a man facing away from the camera, his penis is clearly seen gliding all the way in and all the way back out of her as she rides him, her buttocks and back moving with each stroke. Her face stays in focus the entire time. She looks back over her shoulder toward the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Blowjob POV', 'prompt': 'k3llydw, a woman kneeling in front of a nude man, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and, maintaining eye contact with the viewer, bobs her head all the way down and all the way back up. Her eyes stay locked on the viewer. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'k3llydw', 'name': 'Blowjob Other', 'prompt': 'k3llydw, a woman kneeling beside a nude man seen from the side, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and bobs her head all the way down and all the way back up, glancing toward the camera. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': True}, {'character': 'p@y', 'name': 'Missionary POV', 'prompt': 'p@y, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Missionary Side', 'prompt': 'p@y, a woman lies on her back, her legs spread and raised, knees bent. A man is between her thighs and thrusts into her in a steady, continuous rhythm; his hips meet hers on every stroke and her breasts and thighs jiggle with each impact. His penis is clearly seen gliding all the way in and all the way back out of her vagina. She moans with each thrust, she maintains steady eye contact with the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Her face stays in focus the entire time. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Doggystyle Front Facing', 'prompt': 'p@y, a woman on her knees bent over a bed is facing the camera. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. She maintains steady eye contact with the camera with natural expressions that align with the encounter. Her face stays in focus the entire time. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Doggystyle Side', 'prompt': 'p@y, a woman on her knees bent over a bed seen from the side. A man is behind her having doggystyle sex and thrusts into her in a steady, continuous rhythm; his hips meet her behind on every stroke and her breasts and buttocks jiggle with each impact. Her face is turned toward the camera with natural expressions that align with the encounter. Her face stays in focus the entire time.  Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Cowgirl POV', 'prompt': 'p@y, a woman having cowgirl sex, she is on top of a man, his penis is clearly seen smoothly gliding all the way in and all the way back out of her vagina, her breasts bounce and jiggle as she rides and grinds on him. Her facial expressions show intense pleasure and she maintains eye contact with the viewer. Her face stays in focus the entire time.  Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Cowgirl Backside', 'prompt': 'p@y, a woman having reverse cowgirl sex seen from behind, she is on top of a man facing away from the camera, his penis is clearly seen gliding all the way in and all the way back out of her as she rides him, her buttocks and back moving with each stroke. Her face stays in focus the entire time. She looks back over her shoulder toward the camera. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Blowjob POV', 'prompt': 'p@y, a woman kneeling in front of a nude man, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and, maintaining eye contact with the viewer, bobs her head all the way down and all the way back up. Her eyes stay locked on the viewer. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}, {'character': 'p@y', 'name': 'Blowjob Other', 'prompt': 'p@y, a woman kneeling beside a nude man seen from the side, she grips his penis with one hand and strokes it top to bottom. She wraps her lips around it and bobs her head all the way down and all the way back up, glancing toward the camera. Her other hand rests on his thigh. Handheld camera at the foot of the bed, slight natural sway, no cuts. Soft even lamplight from the left picks out natural skin tones and texture. Audio: her rhythmic breathing and moaning, the sound of skin against skin.', 'validated': False}]

revision = "071"
down_revision = "070"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("ltx_recipe_book")

    op.create_table(
        "ltx_characters",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False, unique=True),
        sa.Column("char_lora", sa.Text(), nullable=False),
        # Per-stage, never flat. Stage 1 generates at half size from noise and stage 2 refines
        # the 2x-upscaled latent; 0.8/1.5 is validated and collapsing them is a different
        # configuration. Per character rather than global so a future character can differ.
        sa.Column("strength_stage_1", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("strength_stage_2", sa.Float(), nullable=False, server_default="1.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )

    op.create_table(
        "ltx_recipes",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("character_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("ltx_characters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        # NULL means "the stack's negative", which is the case for all 24 seeded recipes.
        # An override exists because a pose might one day need one, not because any does.
        sa.Column("negative_prompt", sa.Text(), nullable=True),
        sa.Column("frames", sa.Integer(), nullable=True),
        # Whether a human has watched this and signed it off. Not a quality score — the
        # automated metrics have picked the wrong clip before, repeatedly.
        sa.Column("validated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("character_id", "name", name="uq_ltx_recipe_character_name"),
    )
    op.create_index("ix_ltx_recipes_character_id", "ltx_recipes", ["character_id"])

    # Seed the 24 validated recipes. One time, from the sheet that is now retired.
    #
    # Prompts go in RESOLVED. The sheet stored keys (P.MISS_POV) against a definitions tab;
    # that indirection existed to avoid retyping text in a spreadsheet and buys nothing here,
    # where the console edits the text directly.
    import uuid as _uuid
    conn = op.get_bind()
    ids = {}
    for c in _CHARACTERS:
        cid = _uuid.uuid4()
        ids[c["name"]] = cid
        conn.execute(
            sa.text(
                "INSERT INTO ltx_characters (id, name, char_lora, strength_stage_1, "
                "strength_stage_2) VALUES (:id, :name, :lora, :s1, :s2)"
            ),
            {"id": cid, "name": c["name"], "lora": c["char_lora"],
             "s1": c["strength_stage_1"], "s2": c["strength_stage_2"]},
        )
    for r in _RECIPES:
        conn.execute(
            sa.text(
                "INSERT INTO ltx_recipes (id, character_id, name, prompt, validated) "
                "VALUES (:id, :cid, :name, :prompt, :validated)"
            ),
            {"id": _uuid.uuid4(), "cid": ids[r["character"]], "name": r["name"],
             "prompt": r["prompt"], "validated": r["validated"]},
        )


def downgrade() -> None:
    op.drop_index("ix_ltx_recipes_character_id", table_name="ltx_recipes")
    op.drop_table("ltx_recipes")
    op.drop_table("ltx_characters")
    op.create_table(
        "ltx_recipe_book",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("book", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("book_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=True),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint("id = 1", name="ck_ltx_recipe_book_singleton"),
    )
