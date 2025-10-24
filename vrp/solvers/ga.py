import random
from .solver_base import Solver
from ..core.problem import Problem
from ..core.solution import Solution
from ..core.eval import evaluate


class GASolver(Solver):
    def __init__(self, problem: Problem, seed: int = 42, pop_size: int = 50):
        super().__init__(problem, seed)
        self.pop_size = pop_size


    def solve(self, time_limit_sec: float = 30.0) -> Solution:
        random.seed(self.seed)
        # TODO: implement OX/PMX crossover + 2-opt/insert mutation respecting MD-MV
        # For now, reuse DFASolver init and do trivial selection to keep structure consistent
        from .dfa import DFASolver
        init = DFASolver(self.problem, seed=self.seed, pop_size=self.pop_size)._init_population()
        best = min(init, key=lambda s: evaluate(self.problem, s)[0])
        return best