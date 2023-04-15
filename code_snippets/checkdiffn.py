import time
import matplotlib.pyplot as plt
import SolveKnapsack_21_C3 
import time
import copy
import collections
import queue as Q
import numpy as np
import pandas as pd
import scipy as sp
import itertools
import Random_Instance_Generator
import Instance_Generator_with21
import SolveKnapsack_21_C2
import os
import gurobipy as gp
from gurobipy import GRB
gp.setParam("OutputFlag", 0)
#gp.setParam("MIPGap", 0)
gp.setParam("MIPGap", 1e-6)

np.random.seed(21)
l_list = []
t_list = []
for i in range(5,22):
    print(i)
    # c_i,a_ik, b_k = Instance_Generator_with21.MOP_Generator(n = i,m=1,J = 2, U = 40)

    # while True:
    #     intl = Instance_Generator_with21.MOP_Generator(i,1,2,40)
    
    #     # n,b_k,c_i,a_ik
    #     Z_n= SolveKnapsack_21_C2.SolveKnapsack(intl)
    #     if len(Z_n) >= 3:
    #         break
    # print("solved")
    n = i
    m=1
    J = 2
    U = 40
    inst = "n_"+str(i)+"_m_"+str(m)+"_J_"+str(J)+"_U_"+str(U)+".txt"
    t_list.append(SolveKnapsack_21_C3.SolveKnapsack(filename = inst, method = 2)[0])
    l_list.append(i)
plt.plot(l_list,t_list)
plt.show()