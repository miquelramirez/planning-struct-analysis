from dataclasses import dataclass
from typing import Self

from unified_planning.model import FNode, Fluent, Effect, EffectKind, OperatorKind

from planning_struct_analysis.numeric.structs.expressions import AffineExpression


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
