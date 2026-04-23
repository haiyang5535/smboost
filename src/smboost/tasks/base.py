from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from smboost.harness.state import HarnessState


class TaskGraph(ABC):
    @property
    @abstractmethod
    def node_names(self) -> list[str]:
        """Ordered list of node names to execute (e.g. ['plan', 'execute', 'verify'])."""
        ...

    @abstractmethod
    def get_node_fn(
        self, node_name: str
    ) -> Callable[["HarnessState", "ChatOpenAI"], str]:
        """Return the callable for the given node. Callable takes (state, llm) → output str."""
        ...
