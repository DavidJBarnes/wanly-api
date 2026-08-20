"""Whole-tag matching over a comma-joined tags column.

Tags live as one text blob on both images ("Kelly, Missionary") and jobs ("AR, kelly"), so a
substring match cannot tell where one tag ends and the next begins. Measured on the image repo
2026-08-14, that is not a corner case: `%kelly%` matched 2,057 of 2,788 images -- 74% of the
repo -- because it also caught KellyYoung, KellyBangs and KellyTeacher. Exact Kelly is 824.

This module is the shared implementation, parameterised on the column, so images and jobs agree
on what "has this tag" means.
"""

from sqlalchemy import func


def like_escape(s: str) -> str:
    """Neutralise LIKE metacharacters. Backslash first, or escaping the wildcards would itself
    be re-escaped."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def normalise_tag(tag: str) -> str:
    """Fold a tag to its comparison form: case- and space-insensitive.

    Spaces come out because the vocabulary contains "Big dick" and the stored string is joined
    with ", " -- inconsistent spacing around the commas is the norm, not the exception.
    """
    return tag.strip().lower().replace(" ", "")


def tag_clause(column, tag: str):
    """Match one WHOLE tag inside the comma-joined tags string of `column`.

    Wrapping both sides in commas is what makes the boundaries real: ",kelly," cannot match
    inside ",kellyyoung,". concat() rather than || because it is null-safe, so an untagged row
    simply fails to match instead of poisoning the expression.
    """
    pattern = f"%,{like_escape(normalise_tag(tag))},%"
    return func.concat(",", func.replace(func.lower(column), " ", ""), ",").like(
        pattern, escape="\\"
    )
