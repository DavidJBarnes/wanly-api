"""Parsing the hand-authored LTX recipe sheet.

Every failure here is silent: a mis-parsed sheet still yields a book with the right shape and
the right number of things, and the render that follows looks fine. That is not hypothetical —
a 16-render batch once ran against stale prompts, and the check that missed it compared counts
and one field, both identical in the stale data.
"""

import io
import zipfile

import pytest

from app.ltx_sheet import SheetError, book_sha256, parse_sheet

REAL_SHEET = "/home/david/projects/storyboard/renders/recipe-sheet.ods"


def _ods(sheets: dict[str, list[list[str]]]) -> bytes:
    """Minimal .ods: just enough content.xml for the parser."""
    T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    X = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    parts = [f'<office:document-content xmlns:office="urn:x" xmlns:table="{T}" xmlns:text="{X}">']
    for name, rows in sheets.items():
        parts.append(f'<table:table table:name="{name}">')
        for row in rows:
            parts.append("<table:table-row>")
            for cell in row:
                parts.append(f"<table:table-cell><text:p>{cell}</text:p></table:table-cell>")
            parts.append("</table:table-row>")
        parts.append("</table:table>")
    parts.append("</office:document-content>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("content.xml", "".join(parts))
    return buf.getvalue()


BASIC = {
    "Prompts": [["key", "text"], ["P.A", "a woman walking"], ["N.STD", "blurry, distorted"]],
    "alice": [
        ["id", "label", "source", "Pose One", "Pose Two"],
        ["char_lora", "Char Lora", "ui", "alice_v2", "alice_v2"],
        ["char_s1", "Stage 1", "ui with defaults", "0.8", "0.8"],
        ["char_s2", "Stage 2", "ui with defaults", "1.5", "1.5"],
        ["prompt_positive", "Prompt", "ui with defaults", "P.A", "P.A"],
        ["prompt_negative", "Negative", "ui with defaults", "N.STD", "N.STD"],
        ["validated", "Validated", "", "Yes", "No"],
    ],
}


def test_parses_characters_and_recipes():
    b = parse_sheet(_ods(BASIC))
    assert list(b["characters"]) == ["alice"]
    assert sorted(b["characters"]["alice"]["recipes"]) == ["Pose One", "Pose Two"]
    assert b["definitions"]["P.A"] == "a woman walking"


def test_source_annotations_are_captured_from_their_own_column():
    """The `id` column exists so the machine key is separate from the human label.

    Before it, the source lived inside the label as "Checkpoint (hardcoded)" — and adding that
    annotation broke the parser. Labels must be free to change without breaking anything.
    """
    b = parse_sheet(_ods(BASIC))
    assert b["sources"]["char_lora"] == "ui"
    assert b["sources"]["char_s1"] == "ui with defaults"
    assert b["labels"]["char_s1"] == "Stage 1"


def test_legacy_label_annotation_still_parses():
    legacy = {
        "Prompts": [["key", "text"], ["P.A", "x"]],
        "alice": [["label", "Pose One"], ["char_lora (ui)", "alice_v2"]],
    }
    b = parse_sheet(_ods(legacy))
    assert b["characters"]["alice"]["recipes"]["Pose One"]["char_lora"] == "alice_v2"
    assert b["sources"]["char_lora"] == "ui"


def test_missing_prompts_tab_is_a_named_error_not_a_shrug():
    with pytest.raises(SheetError, match="Prompts"):
        parse_sheet(_ods({"alice": [["id", "label", "source", "Pose"]]}))


def test_not_an_ods_is_a_named_error():
    with pytest.raises(SheetError, match="readable"):
        parse_sheet(b"this is not a zip file")


def test_book_hash_tracks_content_not_formatting():
    """Compare CONTENT, not counts.

    A stale book has the right number of everything, which is exactly how the stale-prompt
    batch went unnoticed. The hash is what makes 'has this actually changed' answerable.
    """
    a = parse_sheet(_ods(BASIC))
    same = parse_sheet(_ods(BASIC))
    assert book_sha256(a) == book_sha256(same)

    edited = {k: [r[:] for r in v] for k, v in BASIC.items()}
    edited["Prompts"][1][1] = "a woman running"      # one word, in shared prompt text
    assert book_sha256(parse_sheet(_ods(edited))) != book_sha256(a)


@pytest.mark.skipif(not __import__("os").path.exists(REAL_SHEET), reason="real sheet not present")
def test_matches_the_real_sheet_and_the_engines_own_output():
    """The port must be faithful, not merely plausible.

    Verified 2026-08-31 against the parser this was ported from: byte-identical book,
    sha256 31dfad3847a8a3f8.
    """
    b = parse_sheet(open(REAL_SHEET, "rb").read())
    assert set(b["characters"]) == {"k3lly2026", "k3llydw", "p@y"}
    assert all(len(c["recipes"]) == 8 for c in b["characters"].values())
    r = b["characters"]["k3lly2026"]["recipes"]["Missionary POV"]
    assert r["char_lora"] == "k3lly2026_v2"
    assert (r["char_s1"], r["char_s2"]) == ("0.8", "1.5")
    # content_lora "none" is not incidental: dropping DR34ML4Y is what removed the motion
    # horror, and it is universal across all sixteen validated recipes.
    assert r["content_lora"] == "none"


def test_first_character_is_mirrored_at_the_top_level():
    """The console falls back to book.recipes / book.sources when there is no character map.

    Dropping these passed every other test in this file — the parser looked fine, the
    characters were all there, and the failure would have surfaced as empty dropdowns in the
    UI against an engine that predates multi-character. Found by deliberately deleting the
    mirror and watching nothing fail.
    """
    b = parse_sheet(_ods(BASIC))
    assert b["character"] == "alice"
    assert b["recipes"] == b["characters"]["alice"]["recipes"]
    assert b["sources"] == b["characters"]["alice"]["sources"]
    assert b["labels"] == b["characters"]["alice"]["labels"]
