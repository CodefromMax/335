import Random_Instance_Generator
import numpy as np
import Bf 
import itertools


np.random.seed(21)

while True:
    C, A, B = Random_Instance_Generator.MOP_Generator(5,1,3,40)

    X = list(itertools.product([0, 1], repeat=5) )
    X_copy = []
    for a, b in zip(A, B):
        for x in X:
            LHS = np. dot(a, x)
            if LHS<= b:
                # print(a, x, LHS, b)
                X_copy. append (x)
        X = X_copy[:]
        X_copy = []
    Z=[]
    for x in X:
        Z.append((np.dot(C[0], x), np.dot(C[1], x)))
    Z = list(set(Z))
    
    #n,b_k,c_i,a_ik
    Z_n= Bf.BF(5,B,C,A)

    if len(Z_n) >= 3:
        print(3)
        print(len(Z))
        print(Z)
        print(Z_n)
        break
