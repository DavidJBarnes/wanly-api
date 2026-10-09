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


async def test_an_archived_set_does_not_collect_orphans(db):
    from datetime import datetime, timezone
    tag = uuid.uuid4().hex[:6]
    old = await _set(db, f"Old{tag}", f"Ann{tag}", [], archived_at=datetime.now(timezone.utc))
    orphan = await _run(db, f"Ann{tag}", 1, [("identity", None)])
    await db.commit()
    assert str(orphan.id) not in await _roles(db, old)


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
