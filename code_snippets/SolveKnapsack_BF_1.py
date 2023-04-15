# imports
import numpy as np
import itertools
import time
import copy
import collections
import queue as Q
import pandas as pd
import scipy as sp
import os


import gurobipy as gp
# Disable gurobi logging
gp.setParam("OutputFlag", 0)
#gp.setParam("MIPGap", 0)
gp.setParam("MIPGap", 1e-6)


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


def get_model(n, m, J, C, A, B):
    """Get the model min_{x in X} alpha_1 * z_1 + alpha_2 * z_2"""
    model = gp.Model(f'z_model')

    # x is a binary decision variable with n dimensions
    x = model.addVars(n, vtype='B', name='x')

    # Define variables for objective
    z = []
    for i in range(J):
        z.append(model.addVar(vtype='I', name=f'z_{i}', obj=1))
        
    # Attach variables to model
    model._x, model._z = x, z    
        
    # Set the z values
    for i in range(J):
        model.addConstr(z[i] == gp.quicksum(C[i][j]*x[j] for j in range(n)))


    # The x in \mathcal X constraint
    for i in range(m):
        model.addConstr(gp.quicksum(A[i][j]*x[j] for j in range(n)) <= B[i][0])

    # The constraints imposed by the region. Since we have defined the objective as 
    # a variable, we can simply modify its upper bound to impose the constraint.
    for i in range(J):
        z[i].ub = 0
        z[i].lb = -gp.GRB.INFINITY

    # Objective
    # alpha_1 and alpha_2 is 1 for now
    model.setObjective(z[0] + z[1], sense=gp.GRB.MINIMIZE)
    
    return model

def lexmin(model, J, first_obj=1, NW=None, SE=None):    
    # set the first and second obj index    
    assert 1 <= first_obj <= 2
    
    z1, z2 = model._z[0], model._z[1]
    if NW == None and SE == None:        
        z1.ub, z1.lb = 0, -gp.GRB.INFINITY        # -infty <= z1 <= 0        
        z2.ub, z2.lb = 0, -gp.GRB.INFINITY        # -infty <= z2 <= 0
    elif NW is not None and SE is not None:        
        z1.ub, z1.lb = SE[0], NW[0]               # NW.x <= z1 <= SE.x                
        z2.ub, z2.lb = NW[1], SE[1]               # SE.y <= z2 <= NW.y
    else:
        raise ValueError('Invalid NW and SE')
        
    # .Obj allows you modify the objective coefficient of a given variable
    # Modify the objective to: 1 x z_1 + 0 x z_2 = z_1 if first_obj == 1
    # Or modify the objective to: 0 x z_1 + 1 x z_2 = z_2 if first_obj == 2
    if first_obj == 1:
        z1.Obj, z2.Obj = 1, 0 
    else:
        z1.Obj, z2.Obj = 0, 1
    
    # Optimize
    model.update()
    model.optimize()
    
    # Checking the model status to verify if the model is solved to optimality
    if model.status == 2:
        first_obj_val = int(np.round(model.objval))
        
        # Update bound and objective coefficients
        if first_obj == 1:
            z1.ub = first_obj_val
            z1.Obj, z2.Obj = 0, 1
        else:
            z2.ub = first_obj_val
            z1.Obj, z2.Obj = 1, 0
              
        # Optimize
        model.update()
        model.optimize()
        
        if model.status == 2:
            second_obj_val = int(np.round(model.objval))
            
            return [first_obj_val, second_obj_val] if first_obj == 1 else [second_obj_val, first_obj_val]
                        
    return None


# Create the model object
def get_weighted_sum_model(n, m, J, C, A, B, region, lam):
    model = gp.Model()

    # x is a binary decision variable with n dimensions
    x = model.addVars(n, vtype='B', name='x')
    
    # Define variables for objective
    z = []
    for i in range(J):
        z.append(model.addVar(vtype='I', name=f'z_{i}'))

    # Attach the vars to the model object
    model._x = x
    model._z = z
        
    # Set the objectives
    for i in range(J):
        model.addConstr(z[i] == gp.quicksum(C[i][j]*x[j] for j in range(n)))


    # The x in \mathcal X constraint
    for i in range(m):
        model.addConstr(gp.quicksum(A[i][j]*x[j] for j in range(n)) <= B[i][0])

    # The constraints imposed by the region. Since we have defined the objective as 
    # a variable, we can simply modify its upper bound to impose the constraint.
    for i in range(J):
        z[i].ub = region[i]
        z[i].lb = -gp.GRB.INFINITY

    # Objective
    model.setObjective(gp.quicksum(lam[i]*z[i] for i in range(J)), 
                       sense=gp.GRB.MINIMIZE)
    
    return model

def get_supernal_z(n, C, model):
  x_var = model._x
  x_sol = [int(np.round(x_var[i].x)) for i in range(n)]

  return np.dot(C, x_sol)

def quicksort(Z):
    if len(Z) <= 1:
        return Z
    else:
        piv = Z[len(Z)//2]
        left = [x for x in Z if x < piv]
        middle = [x for x in Z if x == piv]
        right = [x for x in Z if x > piv]
        return quicksort(left) + middle + quicksort(right)

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

        # BF1: quicksort
        Z_removed_dup = quicksort(Z_removed_dup)

        FoundNDPs = Z_removed_dup.copy()

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
                    FoundNDPs.remove(Z_removed_dup[i])
                    break



    elif method == 2:
        methodName = "RDM"
        # TODO: Read and solve an instance via Rectangle Divison Method (RDM)
        FoundNDPs = []
        n,b_k,c_i,a_ik = read_instance(filename)
        C_int = [cc.astype(int).tolist() for cc in c_i]
        A_int = [a.astype(int).tolist() for a in a_ik]
        b_int = [b.tolist() for b in b_k]
        n = int(n[0])
        first_lex = 0
        second_lex = 0
        m = len(A_int)
        J = len(C_int)
        model = get_model(n, m, J, C_int, A_int, b_int)
        
        first_lex = lexmin(model,J, 1)
        second_lex = lexmin(model,J, 2)

        FoundNDPs.append(first_lex)
        FoundNDPs.append(second_lex)


        # print(first_lex,second_lex)

        Rectangles = [[first_lex,second_lex]]
        while len(Rectangles) != 0:
            picked_rect = Rectangles[0]
            print(picked_rect)
            Rectangles.remove(picked_rect)
            
            R_2 = [(picked_rect[0],(picked_rect[0][1]+picked_rect[1][1])/2),picked_rect[1]]
            # print(R_2[0][0][0])
            # print(R_2[0][1])
            z_1 = lexmin(model,J, first_obj=1,NW = [R_2[0][0][0],R_2[0][1]],SE = R_2[1])

            if z_1 != R_2[1]:
                FoundNDPs.append(z_1)
                print("z_1",z_1)
                Rectangles.append([z_1,R_2[1]])

                #Check if use R_2[0], Z1, -1 , 
            R_3 = [picked_rect[0],(z_1[0]-1,(picked_rect[0][1]+picked_rect[1][1])/2)]
            z_2 = lexmin(model,J,first_obj =2, NW = R_3[0],SE = [R_3[1][0],R_3[1][1]])
            if z_2 != R_3[0]:
                FoundNDPs.append(z_2)
                print("z_2",z_2)
                Rectangles.append([R_3[0],z_2])
        

    elif method == 3:
        methodName = "SPM"
        # TODO: Read and solve an instance via Supernal Method (SPM)
        n,b_k,c_i,a_ik = read_instance(filename)
        C_int = [cc.astype(int).tolist() for cc in c_i]
        A_int = [a.astype(int).tolist() for a in a_ik]
        b_int = [b.tolist() for b in b_k]
        n = int(n[0])
        m = len(A_int)
        J = len(C_int)

        FoundNDPs = []
        z_s = [0]*J   # supernal point of MOP
        Regions = [z_s] # Initiallize the Regions list
        lam = [1]*J # Lambda
        num_region = 1
        # Only called once before the while-loop
        model = get_weighted_sum_model(n, m, J, C_int, A_int, b_int, z_s, lam)

        while len(Regions) != 0 :
            picked_region = Regions[0] # pick a region in Regions list
            # Reuse model. Only update bounds instead of creating a new model.
            for i, r in enumerate(picked_region):
                model._z[i].ub = r
            model.update()
            model.optimize()

            # Checking the model status to verify if the model is solved to optimality (Feasible)
            if model.status == 2: 
                num_region += J
                z_n = get_supernal_z(n, C_int, model)
                FoundNDPs.append(z_n)
                #print("z_n:",z_n)

                for i in Regions:
                    if (z_n <= i).all():
                        Regions.remove(i)
                        for j in range(J):
                            z_new = i.copy()
                            z_new[j] = z_n[j]-1
                            Regions.append(z_new)
                
                if J >= 3:
                    remove_index = []
                    count = 0
                    for i in range(len(Regions)):
                        for j in range(len(Regions)):
                            if all(Regions[i][k] <= Regions[j][k] for k in range(len(Regions[i]))) and i != j:
                                remove_index.append(i)

                    for i in remove_index:
                        Regions.pop(i-count)
                        count += 1           
            
            else:
                #print("removed",Regions[0])
                Regions.remove(Regions[0])
                #print("R",Regions)
    
    solution_time = time.time()-current_time


    # Output result
    ndp_filename = f'{methodName}_NDP_more_{groupNo}.txt'
    summary_filename = f'{methodName}_SUMMARY_more_{groupNo}.txt'

    # TODO: Export NDP and Summary files
    
    curr_dir = os.getcwd() + '/'
    
    
    ndp_array = np.array(FoundNDPs)
    new = np.lexsort((ndp_array[:,1],ndp_array[:,0]))
    ndp_array = ndp_array[np.flip(new)]

    S_array = np.array([solution_time,
                        len(FoundNDPs),
                        num_region])
    # Note: You must set delimiter to '\t' and newline to '\n'. Otherwise, points will be deducted.
    np.savetxt(curr_dir + ndp_filename, ndp_array,delimiter='\t',newline='\n')
    np.savetxt(curr_dir + summary_filename,S_array,delimiter='\t',newline='\n')
    
    # return nondominated_Z
    return ndp_array
   

    

#print(SolveKnapsack("n_5_m_1_J_2_U_40.txt",3))
print(SolveKnapsack("inst_n375_m2_j2.txt",3))