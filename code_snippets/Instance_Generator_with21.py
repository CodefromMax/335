import Random_Instance_Generator as Random_Instance_Generator
import numpy as np
import SolveKnapsack_21_C2


np.random.seed(21)
import numpy as np
import math
np.random.seed(21)
def MOP_Generator(n,m,J,U):
    '''
    n = # of items (x: 1 or 0)
    m = # of bags (b_k is the capacity for bag k)
    J = number of objective function
    U upperbound for c_i and a_i_k 
    '''
    
    #np.random.seed(21)

    c_total_list = []
    a_total_list = []
    b = []

    #Generate c_i_J
    for each_J in range(J):
        c_total_list.append(-np.random.randint(low=1, high=U+1,size = n))
    
    #Generate a_i_k
    for each_k in range(m):
        a_total_list.append(np.random.randint(low=1, high=U+1,size = n))

    #Calculate b_k
    for each_k in range(m):
        local_sum = (math.ceil((1/2)*sum(a_total_list[each_k])))
        b.append(max(max(a_total_list[each_k]), local_sum))
    
    #Write

    inst = "n_"+str(n)+"_m_"+str(m)+"_J_"+str(J)+"_U_"+str(U)

    file = open(inst+".txt","w+")

    file.write(str(n))
    file.write("\n")
    for each_constraint_right_hand in b:
        file.write(str(each_constraint_right_hand))
        file.write("\n")

    for each_coef_list in range(J):
        #Delete , [,]
        coef_temp = ''.join(str(c_total_list[each_coef_list].tolist()).split(','))
        coef_temp = coef_temp.replace('[','')
        coef_temp = coef_temp.replace(']','')
        file.write(coef_temp)
        file.write(" \n")
        counter = 0
    for constraint_coef_list in range(m):
        counter+= 1
        constraint_coef = ''.join(str(a_total_list[constraint_coef_list].tolist()).split(','))
        constraint_coef = constraint_coef.replace('[','')
        constraint_coef = constraint_coef.replace(']','')
        file.write(constraint_coef)
        if counter != m:
            file.write(" \n")
        else:
            file.write(" ")
    file.close()
                

    inst = inst+".txt"
 
    return inst



while True:
    inst = MOP_Generator(50,2,4,40)
    
    # n,b_k,c_i,a_ik
    Z_n= SolveKnapsack_21_C2.SolveKnapsack(inst)

#     if len(Z_n) >= 3:
#         break

# for i in range(22,30):
#     print(i)
#     # c_i,a_ik, b_k = Instance_Generator_with21.MOP_Generator(n = i,m=1,J = 2, U = 40)

#     while True:
#         intl = MOP_Generator(i,1,2,40)
    
#         # n,b_k,c_i,a_ik
#         Z_n= SolveKnapsack_21_C2.SolveKnapsack(intl)
#         if len(Z_n) >= 3:
#             break