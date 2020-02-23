#!/usr/bin/env python3

import argparse
import itertools
import math
import sys

from utils import save_dimacs_cnf, solve
from itertools import combinations
from pdb import set_trace

#python sudoku.py -c .......1.4.........2...........5.4.7..8...3....1.9....3..4..2...5.1........8.6...
#python sudoku.py .......1.4.........2...........5.4.7..8...3....1.9....3..4..2...5.1........8.6...
#python sudoku.py -c .....54..........8.8.19....3....1.6........34....6817.2.4...6.39......2.53.2.....
def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Solve Sudoku problems.')
    parser.add_argument("board", help="A string encoding the Sudoku board, with all rows concatenated,"
                                      " and 0s where no number has been placed yet.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not print any output.')
    parser.add_argument('-c', '--count', action='store_true',
                        help='Count the number of solutions.')
    return parser.parse_args(argv)


def print_solution(solution):
    """ Print a (hopefully solved) Sudoku board represented as a list of 81 integers in visual form. """
    print(f'Solution: {"".join(map(str, solution))}')
    print('Solution in board form:')
    Board(solution).print()

def compute_solution(sat_assignment, variables, size):
    solution = []
    # TODO: Map the SAT assignment back into a Sudoku solution
    for key, value in sat_assignment.items():
        if (value):
            residual = key%9
            if(residual == 0):
                residual =9
            solution.append((residual))
    return solution

def generate_theory(board, verbose):
    """ Generate the propositional theory that corresponds to the given board. """
    size = board.size()
    clauses = []
    variables = [] #from dict to list

    # TODO: DEFINE VARIABLES & CLAUSES
    #LITERAL DEFINITION
    def transform_literal(literal):
        lista = []
        for lit in literal:
            lista.append(lit[0]*81 + lit[1]*9 + lit[2] + 1)
        return lista
    #ONLY ONE VALUE
    def exactly_one(literals):
        clauses = [transform_literal(literals)]

        for C in combinations(literals, 2):
            clauses += [[-1*(C[0][0]*81 + C[0][1]*9 + C[0][2] + 1),
                         -1*(C[1][0]*81 + C[1][1]*9 + C[1][2] + 1)]]
        return clauses

    # Variables definition
    for i in range (9):
        line = []
        for j in range (9):
            column = []
            for k in range (9):
                column.append((i, j, k))
            line.append(column)
        variables.append(line)

    #Constraint #1: a cell has one value only
    for i in range (9):
        for j in range (9):
            clauses += exactly_one(variables[i][j])
    #Constraint #2: one value in row
    for j in range(9):
        for k in range(9):
            clauses += exactly_one([variables[i][j][k] for i in range (9)])
    #Constraint #3: one value in column
    for i in range(9):
        for k in range(9):
            clauses += exactly_one([variables[i][j][k] for j in range(9)])
    #Constraint #4: one vlaue in each 3x3 grid.
    for x in range(3):
        for y in range(3):
            for k in range(9):
                grid_cells = []
                for a in range(3):
                    for b in range(3):
                        grid_cells.append(variables[3 * x + a][3 * y + b][k])
                clauses += exactly_one(grid_cells)

    return clauses, variables, size

def count_number_solutions(board, verbose=False):
    count = 0
    # TODO
    clauses, variables, size = generate_theory(board, verbose)
    sat_assignment=solve_sat_problem(clauses=clauses, filename="theory.cnf",size = size, variables = variables, verbose = verbose)
    # while sat_assignment is not None:
    #     count +=1
    #     clauses.append(constraint(sat_assignment))
    #     sat_assignment=solve_sat_problem(clauses, "theory.cnf",size = size, variables = variables, verbose = verbose)
    #     solution = compute_solution(sat_assignment, variables=variables, size=size)
    #     set_trace()
    while sat_assignment is not None:
        count += 1
        constraint, clauses_extend = [], []
        for i in range (len(sat_assignment)):
            if (sat_assignment[i] is True):
                constraint += [-1*i]
        clauses_extend.append(constraint)
        sat_assignment = solve_sat_problem(clauses=clauses_extend, filename="theory.cnf", size=size, variables= variables, verbose=verbose)

    print(f'Number of solutions: {count}')

def constraint(sat_assignment):
    new_clause = []
    for literal, value in enumerate(sat_assignment):
        if literal == 0:
            continue
        new_literal = -literal if value else literal
        new_clause.append(new_literal)
    return new_clause

def find_one_solution(board, verbose=False):
    clauses, variables, size = generate_theory(board, verbose)
    return solve_sat_problem(clauses, "theory.cnf", size, variables, verbose)

def solve_sat_problem(clauses, filename, size, variables, verbose):
    save_dimacs_cnf(variables, clauses, filename, verbose)
    result, sat_assignment = solve(filename, verbose)
    if result != "SAT":
        if verbose:
            print("The given board is not solvable")
        return None
    solution = compute_solution(sat_assignment, variables, size)
    if verbose:
        print_solution(solution)
    return sat_assignment


class Board(object):
    """ A Sudoku board of size 9x9, possibly with some pre-filled values. """
    def __init__(self, string):
        """ Create a Board object from a single-string representation with 81 chars in the .[1-9]
         range, where a char '.' means that the position is empty, and a digit in [1-9] means that
         the position is pre-filled with that value. """
        size = math.sqrt(len(string))
        if not size.is_integer():
            raise RuntimeError(f'The specified board has length {len(string)} and does not seem to be square')
        self.data = [0 if x == '.' else int(x) for x in string]
        self.size_ = int(size)

    def size(self):
        """ Return the size of the board, e.g. 9 if the board is a 9x9 board. """
        return self.size_

    def value(self, x, y):
        """ Return the number at row x and column y, or a zero if no number is initially assigned to
         that position. """
        return self.data[x*self.size_ + y]

    def all_coordinates(self):
        """ Return all possible coordinates in the board. """
        return ((x, y) for x, y in itertools.product(range(self.size_), repeat=2))

    def print(self):
        """ Print the board in "matrix" form. """
        assert self.size_ == 9
        for i in range(self.size_):
            base = i * self.size_
            row = self.data[base:base + 3] + ['|'] + self.data[base + 3:base + 6] + ['|'] + self.data[base + 6:base + 9]
            print(" ".join(map(str, row)))
            if (i + 1) % 3 == 0:
                print("")  # Just an empty line

def main(argv):
    args = parse_arguments(argv)
    board = Board(args.board)

    if args.count:
        count_number_solutions(board, verbose=False)
    else:
        find_one_solution(board, verbose=not args.quiet)


if __name__ == "__main__":
    main(sys.argv[1:])
