from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from unified_planning.io import PDDLReader
from unified_planning.engines import CompilationKind
from unified_planning.model import Fluent, FNode, OperatorKind
from unified_planning.shortcuts import Compiler

from typing import Self

import numpy as np

@dataclass
class LinearInequality(object):
    a: np.ndarray
    b: float

    def __add__(self, other: Self) -> Self:
        return LinearInequality(self.a + other.a, self.b + other.b)

    def __sub__(self, other: Self) -> Self:
        return LinearInequality(self.a - other.a, self.b - other.b)

    @staticmethod
    def parse_arithmetic_term(expr: FNode, x: list[Fluent]) -> tuple[int|float, int|None]:
        match expr.node_type:
            case OperatorKind.INT_CONSTANT:
                return expr.int_constant_value(), None
            case OperatorKind.REAL_CONSTANT:
                return expr.real_constant_value(), None
            case OperatorKind.TIMES:
                lhs, rhs = expr
                assert lhs.node_type in (OperatorKind.REAL_CONSTANT, OperatorKind.INT_CONSTANT)
                assert rhs.node_type == OperatorKind.FLUENT_EXP
                index_of_var = x.index(rhs)
                match lhs.node_type:
                    case OperatorKind.REAL_CONSTANT:
                        return lhs.real_constant_value(), index_of_var
                    case OperatorKind.INT_CONSTANT:
                        return lhs.int_constant_value(), index_of_var
            case OperatorKind.FLUENT_EXP:
                index_of_var = x.index(expr)
                return 1.0, index_of_var
        raise ValueError(f"Cannot process arithmetic term: {expr}")
        return 0.0, None

    @staticmethod
    def parse_sum_term(expr: FNode, x: list[Fluent]) -> Self:
        equation = LinearInequality(a=np.zeros(len(x)), b=0.0)
        if expr.node_type == OperatorKind.PLUS:
            for sub_term in expr.args:
                value, var = LinearInequality.parse_arithmetic_term(sub_term, x)
                if var is None:
                    equation.b = value
                else:
                    equation.a[var] = value
        else:
            value, var = LinearInequality.parse_arithmetic_term(expr, x)
            if var is None:
                equation.b = value
            else:
                equation.a[var] = value
        return equation


    @staticmethod
    def parse_leq(expr: FNode, x: list[Fluent]) -> Self:
        lhs, rhs = expr.args
        lhs = LinearInequality.parse_sum_term(lhs, x)
        rhs = LinearInequality.parse_sum_term(rhs, x)
        return lhs - rhs



def process_cmd_line() -> Namespace:
    """
    """
    parser = ArgumentParser("percolator: a planner for STL specifications with PDDL 2.1 constraints")

    parser.add_argument("-d", "--domain",
                        type=str,
                        required=True)
    parser.add_argument("-p", "--problem",
                        type=str,
                        required=True)

    opt: Namespace = parser.parse_args()


    return opt

def search_for_linear_inequalities(expr: FNode, x: list[Fluent]) -> list[LinearInequality]:

    inequalities: list[LinearInequality] = []
    print(expr.node_type)

    match expr.node_type:
        case OperatorKind.AND:
            for i, sub_cond_i in enumerate(expr.args):
                inequalities += search_for_linear_inequalities(sub_cond_i, x)
        case OperatorKind.LE:
            inequalities += [LinearInequality.parse_leq(expr, x)]
        case _:
            print("Cannot do anything in nodes of type", expr.node_type)


    return inequalities


def main() -> None:

    opt: Namespace = process_cmd_line()


    reader = PDDLReader()
    problem = reader.parse_problem(opt.domain, opt.problem)

    obj_count: int = 0
    for _ in problem.all_objects:
        obj_count += 1
    print(f"Found {obj_count} objects in problem {opt.problem} of domain {opt.domain}")
    print(problem)
    with Compiler(problem_kind=problem.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
        grounding_result = grounder.compile(problem, CompilationKind.GROUNDING)
        ground_problem = grounding_result.problem
        print(ground_problem)

        # the set of state variables is identified from the initial state
        state_variables: list[Fluent] = []
        for x, v0 in ground_problem.initial_values.items():
            state_variables.append(x)

        print(state_variables)

        for goal_condition in ground_problem.goals:
            equations: list[LinearInequality] = search_for_linear_inequalities(goal_condition, state_variables)

        print(equations)

if __name__ == '__main__':
    main()