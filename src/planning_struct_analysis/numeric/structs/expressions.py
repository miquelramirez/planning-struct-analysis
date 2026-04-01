from dataclasses import dataclass
from typing import Self

import numpy as np
from unified_planning.model import Fluent, FNode, OperatorKind


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
