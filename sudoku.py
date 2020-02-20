#!/usr/bin/env python3

import argparse
import itertools
import math
import sys

from utils import save_dimacs_cnf, solve
from itertools import combinations

#run in terminal: python sudoku.py -c .......1.4.........2...........5.4.7..8...3....1.9....3..4..2...5.1........8.6...

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
    return solution

from pdb import set_trace

def generate_theory(board, verbose):
    """ Generate the propositional theory that corresponds to the given board. """
    size = board.size()
    clauses = []
    variables = {}

    # TODO: DEFINE VARIABLES & CLAUSES

    def transform_literal(literal):
        lista = []
        for lit in literal:
            lista.append(lit[0]*81 + lit[1]*9 + lit[2] + 1)
        return lista

    def exactly_one(literals):

        clauses = [transform_literal(literals)]

        for C in combinations(literals, 2):
            clauses += [[-1*(C[0][0]*81 + C[0][1]*9 + C[0][2] + 1),
                         -1*(C[1][0]*81 + C[1][1]*9 + C[1][2] + 1)]]
        return clauses

    # All the variables we need: each cell has one of the 9 digits
    lits = []

    for i in range (9):
        line = []
        for j in range (9):
            column = []
            for k in range (9):
                column.append((i, j, k))
            line.append(column)
        lits.append(line)

    # Set of constraints #1: a cell has only one value.
    for i in range (9):
        for j in range (9):
            clauses += exactly_one(lits[i][j])
    # Set of constraints #2: each value is used only once in a row.
    for j in range(9):
        for k in range(9):
            clauses += exactly_one([lits[i][j][k] for i in range (9)])
    # Set of constraints #3: each value used exactly once in each column:
    for i in range(9):
        for k in range(9):
            clauses += exactly_one([lits[i][j][k] for j in range(9)])
    # Set of constraints #4: each value used exactly once in each 3x3 grid.
    for x in range(3):
        for y in range(3):
            for k in range(9):
                grid_cells = []
                for a in range(3):
                    for b in range(3):
                        grid_cells.append(lits[3 * x + a][3 * y + b][k])
                clauses += exactly_one(grid_cells)

    variables = lits

    return clauses, variables, size

def count_number_solutions(board, verbose=False):
    count = 0
    # TODO

    print(f'Number of solutions: {count}')


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
