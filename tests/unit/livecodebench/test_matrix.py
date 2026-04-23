import importlib

from benchmarks.livecodebench.matrix import plan_cells


def test_plan_cells_enumerates_full_matrix():
    cells = plan_cells(
        conditions=["C1", "C4"],
        models=["qwen3.5:4b", "qwen3.5:9b"],
        seeds=[0, 1, 2],
    )
    assert len(cells) == 2 * 2 * 3
    assert {(c.condition, c.model, c.seed) for c in cells} == {
        (cond, mdl, seed)
        for cond in ["C1", "C4"]
        for mdl in ["qwen3.5:4b", "qwen3.5:9b"]
        for seed in [0, 1, 2]
    }


def test_cell_id_is_stable():
    cells = plan_cells(conditions=["C1"], models=["qwen3.5:4b"], seeds=[0])
    assert cells[0].cell_id == "C1__qwen3_5_4b__s0"


def test_cell_id_includes_budget_when_present():
    cells = plan_cells(conditions=["C1"], models=["qwen3.5:4b"], seeds=[0], max_tokens=128)
    assert cells[0].cell_id == "C1__qwen3_5_4b__s0__tok128"


def test_matrix_dirs_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("SMBOOST_MATRIX_SHARD_DIR", "/tmp/custom_shards")
    monkeypatch.setenv("SMBOOST_MATRIX_MEMORY_LOG_DIR", "/tmp/custom_memory")
    monkeypatch.setenv("SMBOOST_MATRIX_FINAL_DIR", "/tmp/custom_final")

    import benchmarks.livecodebench.matrix as matrix_mod
    matrix_mod = importlib.reload(matrix_mod)

    assert str(matrix_mod.SHARD_DIR) == "/tmp/custom_shards"
    assert str(matrix_mod.MEMORY_LOG_DIR) == "/tmp/custom_memory"
    assert str(matrix_mod.FINAL_DIR) == "/tmp/custom_final"
