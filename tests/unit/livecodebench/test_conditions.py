from benchmarks.livecodebench.conditions import CONDITIONS


def test_c1_full():
    agent = CONDITIONS["C1"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is True
    assert agent.session_memory is True
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c2_no_grounded_verify():
    agent = CONDITIONS["C2"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is False
    assert agent.session_memory is True
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c3_no_session_memory():
    agent = CONDITIONS["C3"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is True
    assert agent.session_memory is False
    assert agent.shrinkage_enabled is True
    assert agent.scorer_enabled is True


def test_c4_plain_langgraph_retry():
    agent = CONDITIONS["C4"]("qwen3.5:4b", 0)
    assert agent.grounded_verify is False
    assert agent.session_memory is False
    assert agent.shrinkage_enabled is False
    assert agent.scorer_enabled is False
    assert agent.fallback_chain == ["qwen3.5:4b"]


def test_conditions_keys():
    assert set(CONDITIONS.keys()) == {"C1", "C2", "C3", "C4"}
