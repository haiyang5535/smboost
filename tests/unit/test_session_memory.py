from pathlib import Path

from smboost.memory.session import FailureRecord, SessionMemory


def test_record_has_signature():
    mem = SessionMemory()
    rec = mem.record(
        task_id="t1", node="verify", attempt=0,
        error_class="AssertionError",
        error_line='assert double(2) == 4',
        traceback_tail="...",
        first_assertion='assert double(2) == 4',
        prompt_used="prompt",
    )
    assert rec.signature == ("AssertionError", 'assert double(n) == n', 'assert double(2) == 4')


def test_recent_for_task_returns_only_same_task():
    mem = SessionMemory()
    mem.record(task_id="t1", node="verify", attempt=0, error_class="A", error_line="",
               traceback_tail="", first_assertion="", prompt_used="p1")
    mem.record(task_id="t2", node="verify", attempt=0, error_class="B", error_line="",
               traceback_tail="", first_assertion="", prompt_used="p2")
    mem.record(task_id="t1", node="verify", attempt=1, error_class="A", error_line="",
               traceback_tail="", first_assertion="", prompt_used="p1")

    recent = mem.recent_for_task("t1", limit=2)
    assert len(recent) == 2
    assert all(r.task_id == "t1" for r in recent)
    assert [r.attempt for r in recent] == [1, 0]  # newest first


def test_find_similar_cross_task_signature_match():
    mem = SessionMemory()
    mem.record(task_id="t1", node="verify", attempt=0, error_class="TypeError",
               error_line="x + y", traceback_tail="", first_assertion="",
               prompt_used="sort list ints")
    hit = mem.find_similar(task_id="t2", prompt="sort array ints", error_class="TypeError")
    assert hit is not None
    assert hit.task_id == "t1"


def test_find_similar_returns_none_when_no_match():
    mem = SessionMemory()
    mem.record(task_id="t1", node="verify", attempt=0, error_class="ValueError",
               error_line="x", traceback_tail="", first_assertion="",
               prompt_used="completely different subject about cryptography")
    hit = mem.find_similar(task_id="t2", prompt="bake a cake", error_class="TypeError")
    assert hit is None


def test_jsonl_log_written(tmp_path):
    log_path = tmp_path / "mem.jsonl"
    mem = SessionMemory(log_path=log_path)
    mem.record(task_id="t1", node="verify", attempt=0, error_class="A",
               error_line="x", traceback_tail="tb", first_assertion="",
               prompt_used="p")
    mem.close()
    content = log_path.read_text().strip().splitlines()
    assert len(content) == 1
    assert '"task_id": "t1"' in content[0]


def test_hits_counter():
    mem = SessionMemory()
    mem.record(task_id="t1", node="v", attempt=0, error_class="E", error_line="",
               traceback_tail="", first_assertion="", prompt_used="prompt abc")
    assert mem.hits == 0
    mem.recent_for_task("t1")
    assert mem.hits == 1
    mem.find_similar(task_id="t2", prompt="prompt abc xyz", error_class="E")
    assert mem.hits == 2
