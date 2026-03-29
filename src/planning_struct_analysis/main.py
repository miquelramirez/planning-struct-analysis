from argparse import ArgumentParser, Namespace
from pathlib import Path
from unified_planning.io import PDDLReader
from unified_planning.engines import CompilationKind
from unified_planning.shortcuts import Compiler


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
    with Compiler(problem_kind=problem.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
        grounding_result = grounder.compile(problem, CompilationKind.GROUNDING)
        ground_problem = grounding_result.problem
        print(ground_problem)