# imports
import time
import copy
import collections
import queue as Q
import numpy as np
import pandas as pd
import scipy as sp
import itertools

import os
import gurobipy as gp
from gurobipy import GRB

def read_instance(file_name):
    l = []

    inner_l = []

    with open(file_name) as file:
        for line in file:
            l.append(line.rstrip())

    for i in range(len(l)):
        inner_l.append(l[i].split(" "))

    for row in range(len(inner_l)):
        for element in range(len(inner_l[row])):
            inner_l[row][element] = float(inner_l[row][element])

        inner_l[row] = np.array(inner_l[row])

    n = inner_l[0]

    b_k = [inner_l[1]]
    m = 1 

    c_i = []
    a_ik = []

    if n > 1:
        while len(inner_l[m+1]) == 1:
            m += 1
            b_k.append(inner_l[m])

        J = len(inner_l) - (2*m+1)
        for j in range(1,J+1):
            c_i.append(inner_l[m+j])

        for k in range(1,m+1):
            a_ik.append(inner_l[m+J+k])

    return n,b_k,c_i,a_ik

def get_single_objective_model(n,C, A, B, objective=1):
    model = gp.Model(f'z{objective}')

    J = len(C)#Check#######################
    m = len(A)
    # x is a binary decision variable with n dimensions
    x = model.addVars(n, vtype='B', name='x')

    # Define variables for objective
    z = []
    for i in range(J):
        z.append(model.addVar(vtype='I', name=f'z_{i}'))

    model._x, model._z = x, z    
        
    # Set the objectives
    for i in range(J):
        model.addConstr(z[i] == gp.quicksum(C[i][j]*x[j] for j in range(n)))


    # The x in \mathcal X constraint
    for i in range(m):
        model.addConstr(gp.quicksum(A[i][j]*x[j] for j in range(n)) <= B[i])

    # The constraints imposed by the region. Since we have defined the objective as 
    # a variable, we can simply modify its upper bound to impose the constraint.
    for i in range(J):
        z[i].ub = 0
        z[i].lb = -gp.GRB.INFINITY

    # Objective
    model.setObjective(z[objective-1], sense=gp.GRB.MINIMIZE)
    
    return model

# Not give model  models=models

def lexmin(models,first_obj=1, NW=None, SE=None):
    # set the first and second obj index
    assert 1 <= first_obj <= 2    
    second_obj = 2 if first_obj == 1 else 1        
    
    # initialize list to store NDP
    ndp = []
    J = len(models)
    # Reset z bounds based on NW and SE points
    for m in models:
        if NW == None and SE == None:
            for i in range(J):
                m._z[i].ub = 0
                m._z[i].lb = -gp.GRB.INFINITY
        else:
            # NW.x <= z1 <= SE.x
            m._z[0].ub = SE[0]
            m._z[0].lb = NW[0]
            
            # SE.y <= z2 <= NW.y
            m._z[1].ub = NW[1]
            m._z[1].lb = SE[1]
                
                
    # Optimize the first objective
    curr_model = models[first_obj-1]
    curr_model.optimize()
    ndp.append(int(curr_model.objval))
    
    # Optimize the second objective, with a constraint on the first objective
    curr_model = models[second_obj-1]
    curr_model._z[first_obj-1].ub = ndp[-1]
    curr_model.optimize()
    ndp.append(int(curr_model.objval))
    
    return ndp

def SolveKnapsack(filename, method=1):  
    # Dummy group number. Should be replaced by your number
    groupNo = 21
    methodName = ''
    solution_time = 0.0
    current_time = time.time()
  
    if method == 1:
        methodName = "BF"
        # TODO: Read and solve an instance via Brute-Force method   

        n,b_k,c_i,a_ik = read_instance(filename)

        feas_dec_x = []
        all_comb = list(itertools.product([0,1], repeat = int(n)))
    
        #each
        for each_comb_x in range(len(all_comb)):
            meet = 0
            for k in range(len(b_k)):
                if all_comb[each_comb_x]@a_ik[k] <= b_k[k]:
                    meet+=1
            if meet == len(b_k):
                feas_dec_x.append(all_comb[each_comb_x])

        #Find Z 
        Z = []

        for x in range(len(feas_dec_x)):

            Z_g = []
            for objective_coef in range(len(c_i)):
                Z_g.append(feas_dec_x[x] @ c_i[objective_coef])
            Z.append(Z_g)

        #remove duplicated 
        Z_removed_dup = list(set(tuple(z) for z in Z))



        nondominated_Z = Z_removed_dup.copy()

        count = 0
        #dominated = False
        #Starting from all points are non dominated 
        for i in range(len(Z_removed_dup)):

            for each_i in range(len(Z_removed_dup)):
                count = 0
                if each_i == i:
                    continue 
                for j in range(len(c_i)):
            
                    if Z_removed_dup[i][j] >= Z_removed_dup[each_i][j]:
                        count+=1 
                    
                if count == len(c_i):
                    #dominated = True it is a dominated remove it 
                    nondominated_Z.remove(Z_removed_dup[i])
                    break



    elif method == 2:
        methodName = "RDM"
        # TODO: Read and solve an instance via Rectangle Divison Method (RDM)
        FoundNDPs = []
        n,b_k,c_i,a_ik = read_instance(filename)

        
        model_z1 = get_single_objective_model(n,c_i, a_ik, b_k, objective=1)
        model_z2 = get_single_objective_model(n,c_i, a_ik, b_k, objective=2)
        models = [model_z1, model_z2]
        lexmin(models,1)

    solution_time = time.time()-current_time




    

    # Output result
    ndp_filename = f'{methodName}_NDP_{groupNo}.txt'
    summary_filename = f'{methodName}_SUMMARY_{groupNo}.txt'

    # # TODO: Export NDP and Summary files
    # curr_dir = os.getcwd() + '/'
    # ndp_array = np.array(nondominated_Z)
    # S_array = np.array([solution_time,
    #                     len(nondominated_Z),
    #                     0])
    # # Note: You must set delimiter to '\t' and newline to '\n'. Otherwise, points will be deducted.
    # np.savetxt(curr_dir + ndp_filename, ndp_array,delimiter='\t',newline='\n')
    # np.savetxt(curr_dir + summary_filename,S_array,delimiter='\t',newline='\n')
    
    # return nondominated_Z
  
SolveKnapsack("n_5_m_1_J_2_U_40.txt",2)


