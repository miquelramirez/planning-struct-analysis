from argparse import ArgumentParser, Namespace
from unified_planning.io import PDDLReader
from unified_planning.engines import CompilationKind
from unified_planning.model import Fluent
from unified_planning.shortcuts import Compiler

from planning_struct_analysis.numeric.structs.constraints import LinearInequality, AffineEffect, \
    search_for_linear_inequalities


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