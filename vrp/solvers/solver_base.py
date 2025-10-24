from abc import ABC, abstractmethod
from ..core.problem import Problem
from ..core.solution import Solution


class Solver(ABC):
    def __init__(self, problem: Problem, seed: int = 42):
        self.problem = problem
        self.seed = seed

    @abstractmethod
    def solve(self, time_limit_sec: float = 30.0) -> Solution:
        ...