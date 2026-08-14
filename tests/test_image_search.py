"""Image search: whole tags that AND, and a filename fragment that does not.

Two controls, two jobs. `tags` matches a tag in full; `q` matches the S3 key as a substring.

The split exists because one substring search cannot serve both. Filenames need fragments --
images arrive named "00111-1696092597-swapped.png", get referred to by that number in job configs
and notes, and are spread across dated folders, so "which folder was 00111 in" has to be
answerable. Tags need boundaries: measured on production 2026-08-14, `%kelly%` matched 2,057 of
2,788 images (74% of the repo) because it also caught KellyYoung, KellyBangs and KellyTeacher.
Exact Kelly is 824, and Kelly AND Missionary -- previously unaskable -- is 102.

The clause tests run against real Postgres, because the whole point is what the database does
with commas and NULLs, and a rendered-SQL assertion would test the string rather than the
behaviour.
"""

import pytest
from sqlalchemy import func, select

from app.models import ImageMeta
from app.routes.images import image_filter, normalise_tag, search_pattern


class TestSearchPattern:
    def test_wildcards_in_the_query_are_escaped(self):
        # A user typing "100%" must not turn into a match-everything pattern.
        assert search_pattern("100%") == "%100\\%%"

    def test_underscores_are_escaped(self):
        # "_" is a single-character wildcard in LIKE, and filenames are full of them.
        assert search_pattern("a_b") == "%a\\_b%"

    def test_backslash_is_escaped_before_the_wildcards(self):
        # Escaping the wildcards first would then re-escape their backslashes.
        assert search_pattern("a\\b") == "%a\\\\b%"


class TestNormaliseTag:
    def test_folds_case_and_spaces(self):
        # The vocabulary contains "Big dick" and the stored string is joined with ", ", so
        # spacing around commas is inconsistent in practice.
        assert normalise_tag("  Big Dick ") == "bigdick"


@pytest.mark.asyncio
class TestAgainstTheDatabase:
    async def _seed(self, db):
        rows = [
            ImageMeta(path="s3://b/2026-01-01/00111-kelly.png", tags="Kelly, Missionary"),
            ImageMeta(path="s3://b/2026-01-01/00112.png", tags="KellyYoung, Doggystyle"),
            ImageMeta(path="s3://b/2026-01-02/00113.png", tags="KellyBangs"),
            ImageMeta(path="s3://b/2026-01-02/00114.png", tags="Kelly, Doggystyle"),
            ImageMeta(path="s3://b/2026-01-02/00115.png", tags="Gena, Big dick"),
            ImageMeta(path="s3://b/2026-01-03/00116-untagged.png", tags=None),
        ]
        db.add_all(rows)
        await db.flush()

    async def _paths(self, db, q=None, tags=(), exclude=()):
        clauses = image_filter(q, list(tags), list(exclude))
        rows = (await db.execute(select(ImageMeta.path).where(*clauses))).scalars().all()
        return {p.rsplit("/", 1)[1] for p in rows}

    async def test_a_tag_does_not_match_a_longer_tag_that_starts_with_it(self, db):
        # The bug this feature exists for: Kelly returned KellyYoung and KellyBangs too.
        await self._seed(db)
        assert await self._paths(db, tags=["Kelly"]) == {"00111-kelly.png", "00114.png"}

    async def test_tags_are_conjunctive(self, db):
        await self._seed(db)
        assert await self._paths(db, tags=["Kelly", "Missionary"]) == {"00111-kelly.png"}

    async def test_two_tags_no_image_carries_returns_nothing(self, db):
        # There is no OR: two subject pills mean "both on one image", which is usually empty.
        # That is the agreed semantic, and it should be visibly empty rather than quietly unioned.
        await self._seed(db)
        assert await self._paths(db, tags=["Kelly", "Gena"]) == set()

    async def test_matching_ignores_case_and_spacing(self, db):
        await self._seed(db)
        assert await self._paths(db, tags=["  bIg   dick "]) == {"00115.png"}

    async def test_exclude_removes_a_tag(self, db):
        await self._seed(db)
        assert await self._paths(db, tags=["Kelly"], exclude=["Doggystyle"]) == {
            "00111-kelly.png"
        }

    async def test_exclude_keeps_untagged_images(self, db):
        # NOT LIKE against NULL is NULL, which would silently drop every untagged image from any
        # excluded search. They have nothing to exclude, so they pass.
        await self._seed(db)
        assert "00116-untagged.png" in await self._paths(db, q="0011", exclude=["Kelly"])

    async def test_q_matches_the_filename_as_a_fragment(self, db):
        await self._seed(db)
        assert await self._paths(db, q="00111") == {"00111-kelly.png"}

    async def test_q_matches_a_folder_name(self, db):
        # A free side effect of storing the full key, and a useful one.
        await self._seed(db)
        assert len(await self._paths(db, q="2026-01-02")) == 3

    async def test_q_finds_an_untagged_image(self, db):
        # The original reason path matching exists; tags moving to their own control must not
        # take it away.
        await self._seed(db)
        assert await self._paths(db, q="untagged") == {"00116-untagged.png"}

    async def test_q_no_longer_matches_tags(self, db):
        # Deliberate change. Fragment matching is right for filenames and wrong for tags, and
        # "kelly" as free text was returning three quarters of the repo.
        await self._seed(db)
        assert await self._paths(db, q="Missionary") == set()

    async def test_q_and_tags_and_together(self, db):
        await self._seed(db)
        assert await self._paths(db, q="2026-01-02", tags=["Kelly"]) == {"00114.png"}

    async def test_no_criteria_produces_no_clauses(self, db):
        # The route turns this into a 400 rather than serving the entire repo by accident.
        assert image_filter(None, [], []) == []
        assert image_filter("   ", ["  "], []) == []

    async def test_count_and_page_use_the_same_predicate(self, db):
        # If the count and the page query diverge, pagination reports a total it cannot deliver.
        await self._seed(db)
        clauses = image_filter(None, ["Kelly"], [])
        total = (
            await db.execute(select(func.count()).select_from(ImageMeta).where(*clauses))
        ).scalar()
        page = (await db.execute(select(ImageMeta).where(*clauses))).scalars().all()
        assert total == len(page) == 2

    async def test_a_tag_containing_a_wildcard_is_escaped(self, db):
        # "%" in a tag must not turn into match-everything.
        db.add(ImageMeta(path="s3://b/d/pct.png", tags="100%, Kelly"))
        await self._seed(db)
        assert await self._paths(db, tags=["100%"]) == {"pct.png"}


@pytest.mark.asyncio
class TestTagCounts:
    async def test_counts_are_scoped_to_the_current_filter(self, db):
        """The pills must count within the filtered set, not the whole repo.

        That is what makes a dead end visible before it is clicked: with Kelly selected, a tag
        that co-occurs with nothing simply does not appear.
        """
        db.add_all([
            ImageMeta(path="s3://b/d/1.png", tags="Kelly, Missionary"),
            ImageMeta(path="s3://b/d/2.png", tags="Kelly, Doggystyle"),
            ImageMeta(path="s3://b/d/3.png", tags="Gena, Missionary"),
        ])
        await db.flush()

        tag_rows = (
            func.unnest(func.string_to_array(ImageMeta.tags, ","))
            .table_valued("tag")
            .render_derived(name="t")
        )
        tag_expr = func.lower(func.btrim(tag_rows.c.tag))
        clauses = image_filter(None, ["Kelly"], [])
        rows = (
            await db.execute(
                select(tag_expr.label("tag"), func.count().label("count"))
                .select_from(ImageMeta)
                .join(tag_rows, tag_expr.isnot(None))
                .where(*clauses, tag_expr != "")
                .group_by(tag_expr)
            )
        ).all()

        counts = {r.tag: r.count for r in rows}
        assert counts == {"kelly": 2, "missionary": 1, "doggystyle": 1}
        assert "gena" not in counts
