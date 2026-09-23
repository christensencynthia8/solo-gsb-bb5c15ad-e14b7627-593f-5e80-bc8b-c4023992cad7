"""Core algorithm tests for WDRR + aging (service level, deterministic time)."""
import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus, minutes, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler._get_job_or_404(db.get_conn(), job["id"])


def batch(now_minutes):
    return scheduler.run_batch(now=at(now_minutes))


def balances():
    return {t["id"]: t["balance_gpu_min"] for t in scheduler.list_tenants()}


# ---------------------------------------------------------------------------

def test_weighted_round_robin_selection_order(conn):
    submit("a", 1, 30, weight=1.0, when=iso(0))
    submit("b", 1, 30, weight=3.0, when=iso(1))
    submit("b", 1, 30, weight=3.0, when=iso(2))
    submit("b", 1, 30, weight=3.0, when=iso(3))

    r = batch(10)
    # DRR visit order: a first, then b drains across rounds 2 and 3.
    assert r["selection_order"] == ["J000001", "J000002", "J000003", "J000004"]
    assert r["selected_jobs"][0]["tenant_id"] == "a"
    assert r["selected_jobs"][1]["tenant_id"] == "b"
    assert r["used_gpus_after"] == 4
    assert r["free_gpus_after"] == 4
    # a: aging 10 + quantum 60 - cost 30 = 40
    # b: aging 9  + 3 x quantum 180 - 3 x cost 30 = 459
    b = balances()
    assert b["a"] == pytest.approx(40.0)
    assert b["b"] == pytest.approx(459.0)
    # Audit trail includes aging, quantum and dispatch events in order.
    kinds = [e["kind"] for e in r["events"]]
    assert kinds.count("AGING") == 2
    assert kinds.count("DISPATCH") == 4
    assert "QUANTUM" in kinds


def test_deficit_accrues_within_batch_and_carries_across_batches(conn):
    # Two 6-GPU jobs of cost 600; quantum 60/round => 10 rounds to afford.
    submit("a", 6, 100, weight=1.0, when=iso(0))
    submit("b", 6, 100, weight=1.0, when=iso(1))

    r1 = batch(10)
    assert r1["rounds"] == 10
    assert r1["selection_order"] == ["J000001"]
    # a: 10 aging + 600 quantum - 600 cost = 10.
    # b: 9 aging + 9 x 60 quantum (round 10 capacity-blocked mid-round) = 549.
    b1 = balances()
    assert b1["a"] == pytest.approx(10.0)
    assert b1["b"] == pytest.approx(549.0)

    # Repeated fully-blocked batch at the same instant changes nothing.
    r2 = batch(10)
    assert r2["selection_order"] == []
    assert all(e["kind"] != "QUANTUM" for e in r2["events"])
    assert balances()["b"] == pytest.approx(549.0)

    # Once GPUs free up the carried deficit dispatches b with one quantum:
    # 549 + 2 aging + 60 quantum - 600 cost = 11.
    scheduler.complete_job("J000001")
    r3 = batch(12)
    assert r3["selection_order"] == ["J000002"]
    assert balances()["b"] == pytest.approx(11.0)
    assert scheduler.scheduler_state()["used_gpus"] == 6


def test_capacity_constraint_never_exceeded(conn):
    submit("a", 6, 10, weight=1.0, when=iso(0))
    submit("b", 6, 10, weight=1.0, when=iso(1))
    r = batch(5)
    assert r["selection_order"] == ["J000001"]
    state = scheduler.scheduler_state()
    assert state["used_gpus"] == 6
    assert state["free_gpus"] == 2

    # b is capacity-blocked: stays queued, no quantum farming.
    before = balances()["b"]
    r2 = batch(5)
    assert r2["selection_order"] == []
    assert all(e["kind"] != "QUANTUM" for e in r2["events"])
    assert balances()["b"] == before

    # Completing a releases GPUs; b dispatches (carrying earned credit).
    scheduler.complete_job("J000001")
    r3 = batch(7)
    assert r3["selection_order"] == ["J000002"]
    assert scheduler.scheduler_state()["used_gpus"] == 6


def test_oversized_submission_rejected(conn):
    with pytest.raises(scheduler.SchedulerError):
        scheduler.submit_job("a", 99, 10, 1.0)


def test_aging_unblocks_small_job_behind_heavy_tenant(conn):
    # Heavy tenant 'a' saturates capacity first; small tenant 'b' waits.
    submit("a", 8, 10, weight=4.0, when=iso(0))   # cost 80
    submit("b", 1, 60, weight=1.0, when=iso(0))  # cost 60, blocked

    r1 = batch(10)
    assert r1["selection_order"] == ["J000001"]
    assert scheduler.scheduler_state()["used_gpus"] == 8
    # b cannot accrue quantum while capacity-blocked, but aging is reserved.
    b_balance = balances()["b"]
    assert b_balance == pytest.approx(10.0)  # 10 min wait x rate 1

    # After 110 more minutes a finishes; b's aging + quantum covers its job.
    scheduler.complete_job("J000001")
    r2 = batch(120)
    assert r2["selection_order"] == ["J000002"]
    aging = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert aging and aging[0]["amount"] == pytest.approx(110.0)
    # b: 10 + 110 + 60 quantum - 60 cost = 120
    assert balances()["b"] == pytest.approx(120.0)


def test_aging_accrues_even_while_cluster_full(conn):
    # a saturates all 8 GPUs; b's tiny job waits. A full-capacity batch still
    # settles aging so b's compensation accumulates during the wait.
    submit("a", 8, 10, weight=4.0, when=iso(0))
    submit("b", 1, 5, weight=1.0, when=iso(0))
    r1 = batch(10)
    assert r1["selection_order"] == ["J000001"]
    assert balances()["b"] == pytest.approx(10.0)

    # Full cluster: no rounds, no quantum, but another 50 min of aging lands.
    r2 = batch(60)
    assert r2["selection_order"] == []
    assert r2["rounds"] == 0
    assert all(e["kind"] != "QUANTUM" for e in r2["events"])
    aging2 = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert aging2 and aging2[0]["amount"] == pytest.approx(50.0)
    assert balances()["b"] == pytest.approx(60.0)

    # Free the cluster: b dispatches immediately (60 carried + quantum).
    scheduler.complete_job("J000001")
    r3 = batch(60)
    assert r3["selection_order"] == ["J000002"]


def test_aging_window_resets_after_queue_drains(conn):
    submit("a", 1, 10, weight=1.0, when=iso(0))
    r1 = batch(5)
    assert r1["selection_order"] == ["J000001"]
    assert scheduler.get_tenant("a")["aging_settled_at"] is None

    scheduler.complete_job("J000001")
    # A brand new job starts a fresh wait window measured from its arrival.
    submit("a", 1, 100, weight=1.0, when=iso(100))
    r2 = batch(101)
    aging = [e for e in r2["events"] if e["kind"] == "AGING"]
    assert aging and aging[0]["amount"] == pytest.approx(1.0)


def test_cancel_queued_and_running(conn):
    submit("a", 2, 10, weight=1.0, when=iso(0))
    submit("a", 2, 10, weight=1.0, when=iso(1))
    r = batch(2)
    assert r["selection_order"] == ["J000001", "J000002"]
    assert scheduler.scheduler_state()["used_gpus"] == 4

    cancelled = scheduler.cancel_job("J000002")
    assert cancelled["status"] == "CANCELLED"
    assert scheduler.scheduler_state()["used_gpus"] == 2

    with pytest.raises(scheduler.Conflict):
        scheduler.complete_job("J000002")
    with pytest.raises(scheduler.Conflict):
        scheduler.cancel_job("J000002")

    scheduler.complete_job("J000001")
    with pytest.raises(scheduler.NotFound):
        scheduler.complete_job("J000999")


def test_queue_overview_positions(conn):
    submit("a", 1, 10, weight=2.0, when=iso(0))
    submit("a", 1, 10, weight=2.0, when=iso(1))
    submit("b", 1, 10, weight=1.0, when=iso(2))
    q = scheduler.queue_overview()
    assert q["queued_jobs"] == 3
    groups = {g["tenant_id"]: g for g in q["queues"]}
    assert [j["id"] for j in groups["a"]["jobs"]] == ["J000001", "J000002"]
    assert groups["a"]["jobs"][1]["position"] == 2
    assert groups["a"]["weight"] == 2.0


def test_reproducible_selection(conn):
    # Identical state always yields the same order: fixed tenant order and
    # FIFO per tenant, no randomness. 6 x 1 GPU fits the 8-GPU cluster.
    for tid, w in [("a", 1), ("b", 2), ("c", 1)]:
        for i in range(2):
            submit(tid, 1, 20, weight=w, when=iso(i + 0.01 * ord(tid)))
    r = batch(10)
    expected = {
        "a": ["J000001", "J000002"],
        "b": ["J000003", "J000004"],
        "c": ["J000005", "J000006"],
    }
    for tid, jobs in expected.items():
        got = [j["id"] for j in r["selected_jobs"] if j["tenant_id"] == tid]
        assert got == jobs
    assert len(r["selection_order"]) == 6


# ---------------------------------------------------------------------------
# Job dependencies
# ---------------------------------------------------------------------------

def test_dependency_blocks_head_and_consumes_nothing(conn):
    # Cross-tenant: b's job waits on a's job.
    a1 = submit("a", 1, 10, weight=1.0, when=iso(0))
    b1 = submit("b", 1, 10, weight=1.0, when=iso(1), depends_on=[a1["id"]])

    # The dependent reports its blocker (id + current status).
    assert b1["depends_on"] == ["J000001"]
    assert b1["blocked_by"] == [{"id": "J000001", "status": "QUEUED"}]

    r = batch(10)
    # Only the prerequisite runs; b's head is dependency-blocked.
    assert r["selection_order"] == ["J000001"]
    # No QUANTUM and no AGING landed for b while its head was blocked.
    b_events = [e for e in r["events"] if e["tenant_id"] == "b"]
    assert b_events == []
    assert balances()["b"] == 0.0

    # Still blocked while the prerequisite is RUNNING.
    r2 = batch(20)
    assert r2["selection_order"] == []
    assert balances()["b"] == 0.0

    # Completion releases the dependent; next batch dispatches it.
    scheduler.complete_job("J000001")
    r3 = batch(30)
    assert r3["selection_order"] == ["J000002"]
    released = scheduler.get_job("J000002")
    assert released["status"] == "RUNNING"
    assert released["blocked_by"] == []


def test_all_prerequisites_must_complete(conn):
    p1 = submit("a", 1, 5, when=iso(0))
    p2 = submit("a", 1, 5, when=iso(0))
    dep = submit("b", 1, 5, when=iso(0), depends_on=[p2["id"], p1["id"]])
    # Deduplicated and ordered on output.
    assert dep["depends_on"] == ["J000001", "J000002"]

    r = batch(10)
    assert r["selection_order"] == ["J000001", "J000002"]

    scheduler.complete_job("J000001")
    r2 = batch(20)
    assert r2["selection_order"] == []  # J000002 still RUNNING
    detail = scheduler.get_job(dep["id"])
    assert detail["blocked_by"] == [{"id": "J000002", "status": "RUNNING"}]

    scheduler.complete_job("J000002")
    r3 = batch(30)
    assert r3["selection_order"] == ["J000003"]


def test_cancelled_prerequisite_keeps_dependent_blocked(conn):
    p = submit("a", 1, 5, when=iso(0))
    dep = submit("b", 1, 5, when=iso(0), depends_on=[p["id"]])

    scheduler.cancel_job(p["id"])  # prerequisite cancelled while queued
    detail = scheduler.get_job(dep["id"])
    assert detail["status"] == "QUEUED"
    assert detail["blocked_by"] == [{"id": "J000001", "status": "CANCELLED"}]

    r = batch(10)
    assert r["selection_order"] == []
    # No credit of any kind while blocked by a cancelled prerequisite.
    assert all(e["tenant_id"] != "b" for e in r["events"])
    assert balances()["b"] == 0.0

    # And it remains blocked forever on later batches.
    r2 = batch(100)
    assert r2["selection_order"] == []
    assert scheduler.get_job(dep["id"])["blocked_by"] == [
        {"id": "J000001", "status": "CANCELLED"}
    ]

    # /queue marks the same reason.
    q = scheduler.queue_overview()
    job = q["queues"][0]["jobs"][0]
    assert job["blocked_by"] == [{"id": "J000001", "status": "CANCELLED"}]


def test_dependency_on_running_prerequisite_allowed_and_releases(conn):
    p = submit("a", 1, 5, when=iso(0))
    batch(10)
    assert scheduler.get_job(p["id"])["status"] == "RUNNING"
    # Submitting against a RUNNING prerequisite is fine.
    dep = submit("b", 1, 5, when=iso(10), depends_on=[p["id"]])
    assert dep["blocked_by"] == [{"id": "J000001", "status": "RUNNING"}]
    scheduler.complete_job(p["id"])
    r = batch(20)
    assert r["selection_order"] == ["J000002"]


def test_unknown_and_cyclic_dependencies_rejected(conn):
    with pytest.raises(scheduler.SchedulerError, match="unknown depends_on"):
        scheduler.submit_job("a", 1, 5, depends_on=["J999999"])

    p = submit("a", 1, 5, when=iso(0))
    d = submit("b", 1, 5, when=iso(0), depends_on=[p["id"]])
    assert d["depends_on"] == ["J000001"]

    # Job ids are sequential: the next submission is J000003, so depending on
    # it is a self-dependency cycle. Submission must roll back entirely.
    with pytest.raises(scheduler.SchedulerError, match="dependency cycle"):
        scheduler.submit_job("c", 1, 5, depends_on=["J000003"])
    with pytest.raises(scheduler.NotFound):
        scheduler.get_job("J000003")

    # Multi-node cycle in the stored graph is also detected defensively:
    # plant an edge J000001 -> J000002 (closing 1 -> 2 -> 1), then submit a
    # job that depends on a node participating in that cycle.
    db.get_conn().execute(
        "INSERT INTO job_dependencies (job_id, depends_on) VALUES (?, ?)",
        ("J000001", d["id"]),
    )
    with pytest.raises(scheduler.SchedulerError, match="dependency cycle"):
        scheduler.submit_job("c", 1, 5, depends_on=["J000001"])
    with pytest.raises(scheduler.NotFound):
        scheduler.get_job("J000003")


def test_blocked_head_blocks_rest_of_tenant_queue(conn):
    # Strict FIFO: a dependency-blocked head is never skipped to reach a ready
    # job behind it, and the tenant earns no quantum while it waits.
    p = submit("other", 8, 5, when=iso(0))                 # saturates cluster
    submit("a", 1, 5, when=iso(1), depends_on=[p["id"]])  # blocked head
    submit("a", 1, 5, when=iso(2))                         # ready behind it

    r = batch(10)
    assert r["selection_order"] == [p["id"]]
    # Neither of tenant a's jobs ran; a earned no credit via the blocked head.
    assert balances()["a"] == 0.0
    detail = [j for j in scheduler.list_jobs(status="QUEUED")
              if j["tenant_id"] == "a"]
    assert [j["id"] for j in detail] == ["J000002", "J000003"]

    scheduler.complete_job(p["id"])
    r2 = batch(20)
    # Strict FIFO: the formerly blocked head goes first, then the job behind.
    assert r2["selection_order"] == ["J000002", "J000003"]


def test_queue_overview_annotates_blocked_by(conn):
    p = submit("a", 1, 5, when=iso(0))
    submit("b", 1, 5, when=iso(1), depends_on=[p["id"]])
    q = scheduler.queue_overview()
    groups = {g["tenant_id"]: g for g in q["queues"]}
    j = groups["b"]["jobs"][0]
    assert j["depends_on"] == ["J000001"]
    assert j["blocked_by"] == [{"id": "J000001", "status": "QUEUED"}]
    # The prerequisite job itself has no blockers.
    assert groups["a"]["jobs"][0]["blocked_by"] == []


# ---------------------------------------------------------------------------
# Per-tenant concurrent GPU cap
# ---------------------------------------------------------------------------

def test_tenant_gpu_cap_defaults_null_and_is_persisted(conn):
    scheduler.upsert_tenant("a", 1.0)
    assert scheduler.get_tenant("a")["gpu_cap"] is None
    # Job auto-registration also yields an unlimited tenant.
    submit("auto", 1, 5, when=iso(0))
    assert scheduler.get_tenant("auto")["gpu_cap"] is None

    scheduler.upsert_tenant("a", 1.0, gpu_cap=4)
    assert scheduler.get_tenant("a")["gpu_cap"] == 4
    # Omitting the cap keeps it while updating weight; null clears it.
    scheduler.upsert_tenant("a", 2.0)
    t = scheduler.get_tenant("a")
    assert t["gpu_cap"] == 4 and t["weight"] == pytest.approx(2.0)
    scheduler.upsert_tenant("a", 2.0, gpu_cap=None)
    assert scheduler.get_tenant("a")["gpu_cap"] is None

    for bad in (0, -3, 2.5, True):
        with pytest.raises(scheduler.SchedulerError):
            scheduler.upsert_tenant("b", 1.0, gpu_cap=bad)


def test_gpu_cap_blocks_running_saturated_tenant_without_quantum(conn):
    scheduler.upsert_tenant("a", 1.0, gpu_cap=2)
    submit("a", 2, 10, when=iso(0))
    r = batch(10)
    assert r["selection_order"] == ["J000001"]

    # Head would take the tenant to 3 > 2: it stays queued and earns no
    # quantum for the blocked head.
    submit("a", 1, 10, when=iso(11))
    r2 = batch(20)
    assert r2["selection_order"] == []
    assert all(
        e["kind"] != "QUANTUM"
        for e in r2["events"] if e["tenant_id"] == "a"
    )
    # Aging still settles: the tenant is waiting for RUNNING jobs to release.
    assert any(
        e["kind"] == "AGING" and e["tenant_id"] == "a" for e in r2["events"]
    )
    # Same-instant retry cannot farm credit.
    before = balances()["a"]
    r3 = batch(20)
    assert r3["selection_order"] == []
    assert balances()["a"] == pytest.approx(before)


def test_gpu_cap_head_alone_over_cap_never_dispatches(conn):
    scheduler.upsert_tenant("a", 1.0, gpu_cap=2)
    submit("a", 4, 10, when=iso(0))  # exceeds cap even with 0 RUNNING GPUs
    submit("b", 2, 10, when=iso(0))
    r = batch(10)
    assert r["selection_order"] == ["J000002"]
    assert all(
        e["kind"] != "QUANTUM"
        for e in r["events"] if e["tenant_id"] == "a"
    )
    # Even with the whole cluster free, the over-cap head cannot run.
    scheduler.complete_job("J000002")
    assert batch(20)["selection_order"] == []
    # Raising the cap releases it.
    scheduler.upsert_tenant("a", 1.0, gpu_cap=4)
    assert batch(30)["selection_order"] == ["J000001"]


def test_gpu_cap_skips_blocked_tenant_but_others_still_schedule(conn):
    scheduler.upsert_tenant("a", 1.0, gpu_cap=1)
    submit("a", 1, 10, when=iso(0))
    submit("a", 1, 10, when=iso(0))   # blocked once J1 RUNNING (1+1 > 1)
    submit("b", 4, 10, when=iso(0))   # unlimited, fits
    submit("c", 2, 10, when=iso(0))   # unlimited, fits
    r = batch(10)
    # a's blocked head never steals quantum it cannot spend; b and c run.
    assert r["selection_order"] == ["J000001", "J000003", "J000004"]
    assert r["used_gpus_after"] == 7
    tenants = {t["id"]: t for t in r["tenants"]}
    assert tenants["a"]["running_gpus"] == 1
    assert tenants["b"]["running_gpus"] == 4
    assert tenants["c"]["running_gpus"] == 2
    a_quanta = [e for e in r["events"]
                if e["tenant_id"] == "a" and e["kind"] == "QUANTUM"]
    assert len(a_quanta) == 1 and a_quanta[0]["detail"] == "round 1"


def test_gpu_cap_counts_gpus_not_jobs_and_releases_on_finish(conn):
    scheduler.upsert_tenant("a", 1.0, gpu_cap=3)
    # One 3-GPU RUNNING job saturates the cap; a 1-GPU head stays queued.
    submit("a", 3, 10, when=iso(0))
    submit("a", 1, 10, when=iso(0))
    assert batch(10)["selection_order"] == ["J000001"]
    assert batch(11)["selection_order"] == []

    scheduler.complete_job("J000001")
    assert batch(20)["selection_order"] == ["J000002"]

    # Cancelling a RUNNING job releases the cap as well.
    submit("a", 3, 10, when=iso(21))
    assert batch(30)["selection_order"] == []  # J2 occupies 1/3
    scheduler.cancel_job("J000002")
    assert batch(40)["selection_order"] == ["J000003"]
    # Queued-only cancellation changes nothing about the cap (no GPUs held).
    submit("a", 1, 10, when=iso(41))
    j4 = scheduler.list_jobs(status="QUEUED", tenant_id="a")[0]["id"]
    scheduler.cancel_job(j4)
    assert scheduler.get_tenant("a")["running_gpus"] == 3


# ---------------------------------------------------------------------------
# GPU usage window aggregation
# ---------------------------------------------------------------------------

def dispatch(tenant, gpus, minutes, weight=1.0, when=0):
    """Submit and batch-dispatch a job, then return it (RUNNING)."""
    job = submit(tenant, gpus, minutes, weight=weight, when=iso(when))
    scheduler.run_batch(now=at(when))
    return scheduler.get_job(job["id"])


def set_interval(job_id, start_min, end_min=None, status=None):
    """Pin a job's actual RUNNING interval (end_min=None => still RUNNING)."""
    conn = db.get_conn()
    conn.execute(
        "UPDATE jobs SET scheduled_at = ? WHERE id = ?",
        (iso(start_min), job_id),
    )
    if end_min is None:
        conn.execute(
            "UPDATE jobs SET status = 'RUNNING', completed_at = NULL WHERE id = ?",
            (job_id,),
        )
    else:
        conn.execute(
            "UPDATE jobs SET status = ?, completed_at = ? WHERE id = ?",
            (status or "COMPLETED", iso(end_min), job_id),
        )


def usage_tenants(start_min, end_min, tenant_id=None):
    r = scheduler.gpu_usage(at(start_min), at(end_min), tenant_id=tenant_id)
    return r, {t["tenant_id"]: t for t in r["tenants"]}


def test_gpu_usage_finished_jobs_intersected_with_window(conn):
    j1 = dispatch("a", 2, 30)                       # will pin [10, 40)
    j2 = dispatch("b", 4, 60)                       # will pin [30, 90)
    set_interval(j1["id"], 10, 40)
    set_interval(j2["id"], 30, 90)

    r, tenants = usage_tenants(0, 60)
    assert r["start"].endswith("+00:00")
    # a: 2 GPU x 30 min fully inside = 60; b: 4 GPU x 30 min clipped = 120.
    assert tenants["a"]["gpu_minutes"] == pytest.approx(60.0)
    assert tenants["a"]["overlapping_jobs"] == 1
    assert tenants["b"]["gpu_minutes"] == pytest.approx(120.0)
    assert tenants["b"]["overlapping_jobs"] == 1
    assert r["global"]["gpu_minutes"] == pytest.approx(180.0)
    assert r["global"]["overlapping_jobs"] == 2


def test_gpu_usage_clips_on_both_sides(conn):
    j = dispatch("a", 2, 100)
    set_interval(j["id"], 10, 110)
    _, tenants = usage_tenants(20, 60)
    # 2 GPU x 40 min.
    assert tenants["a"]["gpu_minutes"] == pytest.approx(80.0)
    assert tenants["a"]["overlapping_jobs"] == 1


def test_gpu_usage_half_open_boundaries(conn):
    j = dispatch("a", 2, 10)
    set_interval(j["id"], 10, 20)

    # Window ends exactly when the run starts: no overlap (right-open).
    r, tenants = usage_tenants(0, 10)
    assert tenants["a"]["gpu_minutes"] == 0.0
    assert tenants["a"]["overlapping_jobs"] == 0
    assert r["global"]["overlapping_jobs"] == 0

    # Window starts exactly when the run ends: no overlap (left-closed window,
    # run interval also half-open).
    _, tenants = usage_tenants(20, 30)
    assert tenants["a"]["gpu_minutes"] == 0.0
    assert tenants["a"]["overlapping_jobs"] == 0

    # [run_start, run_end) covers the run in full.
    _, tenants = usage_tenants(10, 20)
    assert tenants["a"]["gpu_minutes"] == pytest.approx(20.0)
    assert tenants["a"]["overlapping_jobs"] == 1


def test_gpu_usage_still_running_truncated_at_cutoff(conn):
    j = dispatch("a", 3, 10)
    set_interval(j["id"], 10, None)                 # RUNNING since t=10

    r, tenants = usage_tenants(0, 50)
    assert tenants["a"]["gpu_minutes"] == pytest.approx(120.0)  # 3 x 40
    assert tenants["a"]["overlapping_jobs"] == 1

    # Cutoff before the run started: nothing.
    _, tenants = usage_tenants(0, 10)
    assert tenants["a"]["gpu_minutes"] == 0.0
    assert tenants["a"]["overlapping_jobs"] == 0

    # Window entirely after the start: run assumed ongoing until the cutoff.
    _, tenants = usage_tenants(60, 80)
    assert tenants["a"]["gpu_minutes"] == pytest.approx(60.0)  # 3 x 20


def test_gpu_usage_excludes_queued_jobs(conn):
    run = dispatch("b", 1, 20)
    set_interval(run["id"], 0, 20)
    # Submitted after the batch, never dispatched: stays QUEUED.
    submit("a", 2, 30, when=iso(0))

    r, tenants = usage_tenants(0, 60)
    assert tenants["a"]["gpu_minutes"] == 0.0
    assert tenants["a"]["overlapping_jobs"] == 0
    assert tenants["b"]["gpu_minutes"] == pytest.approx(20.0)
    assert r["global"]["gpu_minutes"] == pytest.approx(20.0)
    assert r["global"]["overlapping_jobs"] == 1


def test_gpu_usage_counts_cancelled_running_not_cancelled_queued(conn):
    ran = dispatch("a", 2, 10)
    set_interval(ran["id"], 10, 25, status="CANCELLED")  # ran then cancelled
    queued = submit("a", 1, 5, when=iso(0))
    scheduler.cancel_job(queued["id"])               # never ran

    _, tenants = usage_tenants(0, 60)
    # Cancelled while RUNNING: actual interval [10, 25) counts.
    assert tenants["a"]["gpu_minutes"] == pytest.approx(30.0)  # 2 x 15
    assert tenants["a"]["overlapping_jobs"] == 1


def test_gpu_usage_lists_tenants_without_overlap_as_zero(conn):
    j = dispatch("a", 2, 10)
    set_interval(j["id"], 0, 10)
    scheduler.upsert_tenant("c", 1.0)

    r, tenants = usage_tenants(0, 60)
    assert set(tenants) == {"a", "c"}
    assert tenants["c"] == {
        "tenant_id": "c", "gpu_minutes": 0.0, "overlapping_jobs": 0}
    assert tenants["a"]["gpu_minutes"] == pytest.approx(20.0)


def test_gpu_usage_tenant_filter(conn):
    ja = dispatch("a", 2, 10)
    jb = dispatch("b", 4, 10)
    set_interval(ja["id"], 0, 10)
    set_interval(jb["id"], 0, 10)

    r, tenants = usage_tenants(0, 60, tenant_id="a")
    assert set(tenants) == {"a"}
    assert tenants["a"]["gpu_minutes"] == pytest.approx(20.0)
    # Global aggregates the filtered scope only.
    assert r["global"]["gpu_minutes"] == pytest.approx(20.0)
    assert r["tenant_id"] == "a"

    with pytest.raises(scheduler.NotFound):
        scheduler.gpu_usage(at(0), at(60), tenant_id="ghost")


def test_gpu_usage_rejects_window_ending_not_after_start(conn):
    with pytest.raises(scheduler.SchedulerError):
        scheduler.gpu_usage(at(30), at(30))
    with pytest.raises(scheduler.SchedulerError):
        scheduler.gpu_usage(at(30), at(10))


def test_gpu_usage_naive_datetimes_treated_as_utc(conn):
    from datetime import datetime

    j = dispatch("a", 2, 10)
    set_interval(j["id"], 10, 40)
    naive = scheduler.gpu_usage(
        datetime(2026, 1, 1, 0, 10), datetime(2026, 1, 1, 1, 0))
    aware, _ = usage_tenants(10, 60)
    assert naive["global"] == aware["global"]
