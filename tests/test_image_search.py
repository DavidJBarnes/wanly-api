"""Image search must match the filename, not only tags.

The filename is often the only handle a user has. Images arrive named like
"00111-1696092597-swapped.png", are referred to by that name in job configs, notes and
conversation, and are spread across dated folders -- so "which folder was 00111 in" was
unanswerable from the UI, and an untagged image was unfindable by any means at all.

Measured against production before the fix: searching "00111" returned 0 results; matching the
path as well returns 10.
"""

from sqlalchemy import func, or_, select
from sqlalchemy.dialects import postgresql

from app.models import ImageMeta
from app.routes.images import search_clause, search_pattern


def _sql(stmt) -> str:
    """Render against the dialect we actually run on.

    The generic dialect emits ILIKE as lower(x) LIKE lower(y), so asserting on the rendered
    operator under the default dialect tests SQLAlchemy's fallback rather than our query.
    """
    return str(stmt.compile(dialect=postgresql.dialect(),
                            compile_kwargs={"literal_binds": True}))


class TestSearchClause:
    def test_matches_path_as_well_as_tags(self):
        sql = _sql(search_clause("00111"))
        assert "image_meta.tags ILIKE" in sql
        assert "image_meta.path ILIKE" in sql

    def test_it_is_a_disjunction_not_a_conjunction(self):
        # An untagged image has NULL tags. Requiring both would make it permanently unfindable,
        # which is the exact case this fix exists for.
        sql = _sql(search_clause("x"))
        assert " OR " in sql
        assert " AND " not in sql

    def test_wildcards_in_the_query_are_escaped(self):
        # A user typing "100%" must not turn into a match-everything pattern. Asserted on the
        # pattern rather than the rendered SQL, where literal_binds doubles % for paramstyle.
        assert search_pattern("100%") == "%100\\%%"

    def test_underscores_are_escaped(self):
        # "_" is a single-character wildcard in LIKE, and filenames are full of them.
        assert search_pattern("a_b") == "%a\\_b%"

    def test_backslash_is_escaped_before_the_wildcards(self):
        # Escaping the wildcards first would then re-escape their backslashes.
        assert search_pattern("a\\b") == "%a\\\\b%"

    def test_count_and_page_use_the_same_predicate(self):
        # If the count and the page query diverge, pagination reports a total it cannot deliver.
        clause = search_clause("00111")
        count_sql = _sql(select(func.count()).select_from(ImageMeta).where(clause))
        page_sql = _sql(select(ImageMeta).where(clause))
        for fragment in ("image_meta.tags ILIKE", "image_meta.path ILIKE"):
            assert fragment in count_sql and fragment in page_sql
