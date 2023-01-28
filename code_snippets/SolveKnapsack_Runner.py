import SolveKnapsack_GroupNo as solver

filename = 'path/to/file'
methods = [1, 2, 3, 4, 5]

# The method argument can take integer values from 1 to 5
for method in methods:
  solver.SolveKnapsack(filename, method)