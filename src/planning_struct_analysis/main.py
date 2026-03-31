from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from unified_planning.io import PDDLReader
from unified_planning.engines import CompilationKind
from unified_planning.model import Fluent, FNode, OperatorKind, Effect, EffectKind
from unified_planning.shortcuts import Compiler

from typing import Self

import numpy as np

@dataclass
class AffineExpression(object):
    """
    An affine expression
    """
    a: np.ndarray
    b: float

    def __add__(self, other: Self) -> Self:
        return AffineExpression(self.a + other.a, self.b + other.b)

    def __sub__(self, other: Self) -> Self:
        return AffineExpression(self.a - other.a, self.b - other.b)

    @classmethod
    def from_var(cls, y, x: list[Fluent]) -> Self:
        a = np.zeros(len(x))
        a[y] = 1.0
        return AffineExpression(a=a, b=0.0)

    @classmethod
    def from_constant(cls, c, x: list[Fluent]) -> Self:
        a = np.zeros(len(x))
        return AffineExpression(a=a, b=c)

    @classmethod
    def parse_arithmetic_term(cls, expr: FNode, x: list[Fluent]) -> tuple[int|float, int|None]:
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
            case OperatorKind.BOOL_CONSTANT:
                return 1 if expr.bool_constant_value() else 0, None
        raise ValueError(f"Cannot process arithmetic term: {expr}")
        return 0.0, None

    @classmethod
    def parse_term(cls, expr: FNode, x: list[Fluent]) -> Self:
        expression = AffineExpression(a=np.zeros(len(x)), b=0.0)
        if expr.node_type == OperatorKind.PLUS:
            for sub_term in expr.args:
                value, var = AffineExpression.parse_arithmetic_term(sub_term, x)
                if var is None:
                    expression.b = value
                else:
                    expression.a[var] = value
        elif expr.node_type == OperatorKind.MINUS:
            for sub_term in expr.args:
                value, var = AffineExpression.parse_arithmetic_term(sub_term, x)
                if var is None:
                    expression.b = value
                else:
                    expression.a[var] = -value
        else:
            value, var = AffineExpression.parse_arithmetic_term(expr, x)
            if var is None:
                expression.b = value
            else:
                expression.a[var] = value
        return expression

@dataclass
class LinearInequality(object):
    """
    An equation of the form \sum_{i=1}^n a_i x_i + b <= 0
    """
    xi: AffineExpression

    @classmethod
    def parse_leq(cls, expr: FNode, x: list[Fluent]) -> Self:
        lhs, rhs = expr.args
        xi_l = AffineExpression.parse_term(lhs, x)
        xi_r = AffineExpression.parse_term(rhs, x)
        return [LinearInequality(xi=xi_l - xi_r)]

    @classmethod
    def parse_eq(cls, expr: FNode, x: list[Fluent]) -> Self:
        lhs, rhs = expr.args
        xi_l = AffineExpression.parse_term(lhs, x)
        xi_r = AffineExpression.parse_term(rhs, x)
        return [LinearInequality(xi=xi_l - xi_r), LinearInequality(xi=xi_r - xi_l)]

    @classmethod
    def parse_fluent(cls, expr: FNode, x: list[Fluent]) -> Self:
        index_x = x.index(expr)
        xi_l = AffineExpression.from_var(index_x, x)
        xi_r = AffineExpression.from_constant(1, x)
        return [LinearInequality(xi=xi_l - xi_r), LinearInequality(xi=xi_r - xi_l)]


@dataclass
class AffineEffect(object):
    x_plus: int
    xi: AffineExpression

    @classmethod
    def parse(cls, eff: Effect, x: list[Fluent]) -> Self | None:
        aff_x = x.index(eff.fluent)
        eff_expr = eff.value
        match eff.kind:
            case EffectKind.INCREASE:
                eff_rhs = AffineExpression.from_var(aff_x, x) + AffineExpression.parse_term(eff_expr, x)
                return AffineEffect(x_plus=aff_x, xi=eff_rhs)
            case EffectKind.DECREASE:
                eff_rhs = AffineExpression.from_var(aff_x, x) - AffineExpression.parse_term(eff_expr, x)
                return AffineEffect(x_plus=aff_x, xi=eff_rhs)
            case EffectKind.ASSIGN:
                eff_rhs = AffineExpression.parse_term(eff_expr, x)
                return AffineEffect(x_plus=aff_x, xi=eff_rhs)
        return None


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

    match expr.node_type:
        case OperatorKind.AND:
            for i, sub_cond_i in enumerate(expr.args):
                inequalities += search_for_linear_inequalities(sub_cond_i, x)
        case OperatorKind.LE:
            inequalities += LinearInequality.parse_leq(expr, x)
        case OperatorKind.EQUALS:
            inequalities += LinearInequality.parse_eq(expr, x)
        case OperatorKind.FLUENT_EXP:
            inequalities += LinearInequality.parse_fluent(expr, x)
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

        print("goal:")

        for index, eq in enumerate(equations):
            print(f"{index}. {eq}")

        preconditions: list[list[LinearInequality]] = []
        effects: list[list[AffineEffect]] = []
        for action in ground_problem.actions:
            action_equations: list[LinearInequality] = []
            for cond in action.preconditions:
                action_equations += search_for_linear_inequalities(cond, state_variables)
            print(action.name)
            print("precondition:")
            for index, eq in enumerate(action_equations):
                print(f"{index}. {eq}")
            preconditions += [action_equations]
            eff_list: list[AffineEffect] = []
            for eff in action.effects:
                e = AffineEffect.parse(eff, state_variables)
                eff_list += [e]
            effects += [eff_list]
            print("effects:")
            for index, eff_expr in enumerate(effects[-1]):
                print(f"{index}. {state_variables[eff_expr.x_plus]}+ := {eff_expr.xi}")


if __name__ == '__main__':
    main()