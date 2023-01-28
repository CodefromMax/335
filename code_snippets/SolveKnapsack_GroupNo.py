# Python imports
import time
import copy
import collections
import queue as Q
import numpy as np
import pandas as pd
import scipy as sp

import gurobipy as gp
from gurobipy import GRB


def SolveKnapsack(filename, method):  
  # Dummy group number. Should be replaced by your number
  groupNo = 1212
  methodName = ''

  if method == 1:
    methodName = "BF"
    # TODO: Read and solve an instance via Brute-Force method
    
  elif method == 2:
    methodName = "RDM"
    # TODO: Read and solve an instance via Rectangle Divison Method (RDM)

  elif method == 3:
    methodName = "SPM"
    # TODO: Read and solve an instance via Supernal Method (SPM)

  elif method == 4:
    methodName = "COMP_2D"
    # TODO: Read and solve an instance via RDM or SPM designed for competition with J=2

  elif method == 5:
    methodName = "COMP_3D"
    # TODO: Read and solve an instance via RDM or SPM designed for competition with J=3
    

  # Output result
  ndp_filename = f'{methodname}_NDP_{groupNo}.txt'
  summary_filename = f'{methodname}_SUMMARY_{groupNo}.txt'

  # TODO: Export NDP and Summary files

  return
  
  