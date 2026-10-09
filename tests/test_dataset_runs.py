"""A dataset's run history (wanly-console#647): training is reached through datasets only.

GET /datasets/{id}/runs lists every run on a set's page with its role there; GET
/training/{id}/home names the page an old /training link should open.
"""
import uuid

import pytest

from tests.test_living_datasets import _http, _set

pytestmark = pytest.mark.asyncio


async def _run(db, character, version, links, mode="solo"):
    from app.models import TrainingJob, TrainingRunDataset
    job = TrainingJob(character=character, trigger=character.lower(), version=version,
                      status="completed", dataset_images=[], config={"mode": mode})
    db.add(job)
    await db.flush()
    for i, (kind, ds) in enumerate(links):
        db.add(TrainingRunDataset(training_job_id=job.id, group_index=i, kind=kind,
                                  dataset_id=ds.id if ds else None,
                                  dataset_name=ds.name if ds else "gone", images=[]))
    await db.flush()
    return job


async def _roles(db, ds):
    r = await _http(db, "get", f"/datasets/{ds.id}/runs")
    assert r.status_code == 200, r.text
    return {x["job_id"]: (x["role"], (x["pair"] or {}).get("dataset_id")) for x in r.json()}


async def test_solo_pair_and_orphan_runs_land_on_the_right_pages(db):
    tag = uuid.uuid4().hex[:6]
    me = await _set(db, f"Me{tag}", f"Me{tag}", [])
    jo = await _set(db, f"Jo{tag}", f"Jo{tag}", [])
    pair = await _set(db, f"MeJo{tag}", f"MeJo{tag}", [], kind="composition")
    solo = await _run(db, f"Jo{tag}", 1, [("identity", jo)])
    duo = await _run(db, f"MeJo{tag}", 1, [("identity", me), ("identity", jo),
                                          ("composition", pair)], mode="pair")
    orphan = await _run(db, f"Me{tag}", 1, [("identity", None)])
    await db.commit()

    # The pair run's home is its composition set; its members list it with a link there.
    assert await _roles(db, pair) == {str(duo.id): ("home", None)}
    assert await _roles(db, jo) == {str(solo.id): ("home", None),
                                   str(duo.id): ("pair_member", str(pair.id))}
    # Every set the orphan trained on is gone: it lands on its character's living set.
    assert await _roles(db, me) == {str(duo.id): ("pair_member", str(pair.id)),
                                   str(orphan.id): ("orphan", None)}

    homes = {}
    for job in (solo, duo, orphan):
        r = await _http(db, "get", f"/training/{job.id}/home")
        homes[job.id] = r.json()["dataset_id"]
    assert homes == {solo.id: str(jo.id), duo.id: str(pair.id), orphan.id: str(me.id)}


async def test_an_orphan_with_no_living_set_still_lands_somewhere(db):
    """Payton, 2026-10-09: the living set was deleted after the backfill, leaving only
    archived version sets. A run named after its set lands on that set; otherwise on the
    character's newest set, archived or not. No run may become unreachable."""
    from datetime import datetime, timedelta, timezone
    tag = uuid.uuid4().hex[:6]
    now = datetime.now(timezone.utc)
    v1 = await _set(db, f"Pay{tag} v1", f"Pay{tag}", [], archived_at=now, age_days=3)
    v2 = await _set(db, f"Pay{tag} v2", f"Pay{tag}", [], archived_at=now, age_days=1)
    synth = await _run(db, f"Pay{tag}-Synthetic", 1, [("identity", None)])
    from app.models import TrainingRunDataset
    from sqlalchemy import update
    await db.execute(update(TrainingRunDataset)
                     .where(TrainingRunDataset.training_job_id == synth.id)
                     .values(dataset_name=v1.name))
    plain = await _run(db, f"Pay{tag}", 1, [("identity", None)])
    await db.commit()
    assert (await _roles(db, v1)).get(str(synth.id)) == ("orphan", None)
    assert (await _roles(db, v2)).get(str(plain.id)) == ("orphan", None)
    r = await _http(db, "get", f"/training/{plain.id}/home")
    assert r.json()["dataset_id"] == str(v2.id)

    # A living set, when there is one, wins over the archived ones.
    living = await _set(db, f"Pay{tag}", f"Pay{tag}", [])
    await db.commit()
    assert (await _roles(db, living)).get(str(plain.id)) == ("orphan", None)
    assert str(plain.id) not in await _roles(db, v2)


async def test_runs_are_newest_first_and_unknown_sets_are_404(db):
    tag = uuid.uuid4().hex[:6]
    jo = await _set(db, f"Jo{tag}", f"Jo{tag}", [])
    first = await _run(db, f"Jo{tag}", 1, [("identity", jo)])
    second = await _run(db, f"Jo{tag}", 2, [("identity", jo)])
    from datetime import datetime, timedelta, timezone
    first.created_at = datetime.now(timezone.utc) - timedelta(days=1)
    await db.commit()
    r = await _http(db, "get", f"/datasets/{jo.id}/runs")
    assert [x["job_id"] for x in r.json()] == [str(second.id), str(first.id)]
    assert (await _http(db, "get", f"/datasets/{uuid.uuid4()}/runs")).status_code == 404
