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
    solution_time = time.time()-current_time
    
    


    

    # Output result
    ndp_filename = f'{methodName}_NDP_{groupNo}.txt'
    summary_filename = f'{methodName}_SUMMARY_{groupNo}.txt'

    # TODO: Export NDP and Summary files
    curr_dir = os.getcwd() + '/'
    ndp_array = np.array(nondominated_Z)
    S_array = np.array([solution_time,
                        len(nondominated_Z),
                        0])
    # Note: You must set delimiter to '\t' and newline to '\n'. Otherwise, points will be deducted.
    np.savetxt(curr_dir + ndp_filename, ndp_array,delimiter='\t',newline='\n')
    np.savetxt(curr_dir + summary_filename,S_array,delimiter='\t',newline='\n')
    
    return nondominated_Z
  
SolveKnapsack("n_5_m_1_J_2_U_40.txt")


