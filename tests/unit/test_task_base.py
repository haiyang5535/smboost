import pytest
from smboost.tasks.base import TaskGraph

def test_taskgraph_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        TaskGraph()

def test_concrete_subclass_requires_node_names_and_get_node_fn():
    class Incomplete(TaskGraph):
        pass
    with pytest.raises(TypeError):
        Incomplete()

def test_concrete_subclass_instantiates_when_complete():
    class Minimal(TaskGraph):
        @property
        def node_names(self):
            return ["step"]

        def get_node_fn(self, node_name):
            return lambda state, llm: "output"

    agent = Minimal()
    assert agent.node_names == ["step"]
    assert agent.get_node_fn("step")(None, None) == "output"
