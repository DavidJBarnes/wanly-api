"""Parse the hand-authored LTX recipe sheet (.ods) into the recipe book.

The sheet is the source of truth for what a recipe IS, and **David authors it by hand**.
Two consequences that this module exists to respect:

* **Never regenerate the sheet.** Regenerating it from code once silently overwrote a
  hand-edited prompt; the graph hash caught it within minutes, but only because something
  was watching. Reading is the only direction.
* **Parse defensively.** Labels acquire annotations — adding "(hardcoded)" to a column-A
  label broke the parser once. The sheet now leads with an `id` column precisely so the
  machine key is separate from the human label, and labels can change freely.

Ported from storyboard's `recipes/sheet_to_yaml.py`. The parsing lives HERE now: wanly-api
owns the recipe book, and an engine that cannot look a recipe up cannot look up a stale one.
That is the point — two copies of recipes.json once shipped a 16-render batch against stale
prompts, and the check that missed it compared counts rather than content.
"""

import hashlib
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
X = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

# A cell repeated more than this is ODS padding out to the sheet edge, not real data.
_REPEAT_IS_PADDING = 40


class SheetError(ValueError):
    """The upload is not a recipe sheet we can read."""


def _read_sheets(data: bytes) -> dict[str, list[list[str]]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ET.fromstring(z.read("content.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        raise SheetError(f"not a readable .ods file: {type(e).__name__}: {e}") from e

    sheets: dict[str, list[list[str]]] = {}
    for sh in root.iter(f"{{{T}}}table"):
        rows: list[list[str]] = []
        for row in sh.findall(f".//{{{T}}}table-row"):
            cells: list[str] = []
            for c in row.findall(f"{{{T}}}table-cell"):
                rep = int(c.get(f"{{{T}}}number-columns-repeated", "1"))
                txt = " ".join(
                    "".join(p.itertext()).strip() for p in c.findall(f"{{{X}}}p")
                )
                if rep > _REPEAT_IS_PADDING:
                    rep = 1
                cells.extend([txt.strip()] * rep)
            while cells and cells[-1] == "":
                cells.pop()
            rows.append(cells)
        sheets[sh.get(f"{{{T}}}name")] = rows
    return sheets


def _one_character(rows: list[list[str]]) -> dict[str, Any]:
    grid = [r for r in rows if r]
    if not grid:
        raise SheetError("character tab is empty")
    header = grid[0]

    # Leading `id | label | source` columns are the current layout. The legacy layout put
    # the source in the label as "Checkpoint (hardcoded)" — which is exactly the coupling
    # that broke the parser, and why the id column exists.
    has_id = header[0].strip().lower() == "id"
    off = 3 if has_id else 1
    names = header[off:]

    params: dict[str, list[str]] = {}
    sources: dict[str, str] = {}
    labels: dict[str, str] = {}
    for r in grid[1:]:
        if not r or not r[0]:
            continue
        if has_id:
            key = r[0].strip()
            labels[key] = r[1] if len(r) > 1 else ""
            sources[key] = r[2] if len(r) > 2 else ""
        else:
            m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", r[0])
            key = (m.group(1) if m else r[0]).strip()
            labels[key] = key
            sources[key] = m.group(2).strip() if m else ""
        params[key] = r[off:]

    out: dict[str, Any] = {"sources": sources, "labels": labels, "recipes": {}}
    for i, name in enumerate(names):
        def g(key: str, default: str = "") -> str:
            v = params.get(key, [])
            return v[i] if i < len(v) else default

        out["recipes"][name] = {
            "checkpoint": g("checkpoint"),
            "char_lora": g("char_lora"),
            "char_s1": g("char_s1"),
            "char_s2": g("char_s2"),
            "content_lora": g("content_lora"),
            "distill": g("distill"),
            "prompt": g("prompt_positive"),
            "negative": g("prompt_negative"),
            "guidance": g("guidance"),
            "steps": g("steps"),
            "frames": g("frames"),
            "resolution": g("resolution"),
            "test_images": [s.strip() for s in g("input_image").split(",") if s.strip()],
            "validated": g("validated"),
        }
    return out


def parse_sheet(data: bytes) -> dict[str, Any]:
    """Bytes of a .ods -> the recipe book. Pure: reads, never writes."""
    sheets = _read_sheets(data)
    if "Prompts" not in sheets:
        raise SheetError(
            f"no 'Prompts' tab — found {sorted(sheets)}. That tab holds the shared "
            "prompt/negative/guidance text every recipe refers to by key."
        )
    definitions = {
        r[0]: r[1] for r in sheets["Prompts"] if len(r) >= 2 and r[0] != "key"
    }
    characters = [k for k in sheets if k != "Prompts"]
    if not characters:
        raise SheetError("no character tabs — a sheet needs at least one besides 'Prompts'")

    book: dict[str, Any] = {"definitions": definitions, "characters": {}}
    for ch in characters:
        book["characters"][ch] = _one_character(sheets[ch])

    # Back-compat: the first character also sits at the top level, which is the shape the
    # engine served and the console already reads.
    first = book["characters"][characters[0]]
    book.update({
        "character": characters[0],
        "sources": first["sources"],
        "labels": first["labels"],
        "recipes": first["recipes"],
    })
    return book


def book_sha256(book: dict[str, Any]) -> str:
    """Content hash of the whole book.

    Compare CONTENT, not counts. A stale book has the right number of everything — that is
    precisely how a 16-render batch against stale prompts went unnoticed.
    """
    import json
    return hashlib.sha256(json.dumps(book, sort_keys=True).encode()).hexdigest()
