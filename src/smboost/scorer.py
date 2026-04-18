from __future__ import annotations
from difflib import SequenceMatcher
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama
    from smboost.harness.state import HarnessState


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class RobustnessScorer:
    # threshold=0.6: self-consistency cutoff from Wang et al. 2023.
    # Configurable via HarnessAgent(scorer_threshold=...) for per-task recalibration.
    def __init__(self, n_samples: int = 3, threshold: float = 0.6):
        self.n_samples = n_samples
        self.threshold = threshold

    def score(
        self,
        node_fn: Callable,
        state: HarnessState,
        llm: ChatOllama,
    ) -> tuple[str, float]:
        """Run node_fn n_samples times; return (centroid_output, confidence).

        confidence = mean pairwise LCS similarity across all samples.
        centroid = output with highest mean similarity to all others; first on tie.
        """
        outputs = [node_fn(state, llm) for _ in range(self.n_samples)]
        n = len(outputs)
        if n == 1:
            return outputs[0], 1.0

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sims = [_similarity(outputs[i], outputs[j]) for i, j in pairs]
        confidence = sum(sims) / len(sims)

        mean_sims = [
            sum(_similarity(outputs[i], outputs[j]) for j in range(n) if j != i) / (n - 1)
            for i in range(n)
        ]
        best_idx = mean_sims.index(max(mean_sims))
        return outputs[best_idx], confidence
