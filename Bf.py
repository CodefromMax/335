import numpy as np

#n = 0
#b = []

def read_instance(file_name):
    l = []

    inner_l = []
    #with open("n_5_m_1_J_2_U_40.txt") as file:
    #with open("/Users/max/Desktop/335-1/inst_n375_m2_j2.txt") as file:
    with open(file_name) as file:

        
        for line in file:
            #print(line.rstrip())
            l.append(line.rstrip())
    #print(l)



    for i in range(len(l)):
        inner_l.append(l[i].split(" "))

    for row in range(len(inner_l)):
        for element in range(len(inner_l[row])):
            inner_l[row][element] = float(inner_l[row][element])

        inner_l[row] = np.array(inner_l[row])

    #print(inner_l)
    n = inner_l[0]

    b_k = [inner_l[1]]
    m = 1 

    c_i = []
    a_ik = []

    if n > 1:
        while len(inner_l[m+1]) == 1:
            m += 1
            b_k.append(inner_l[m])
        #print("b:",b_k)

        J = len(inner_l) - (2*m+1)
        for j in range(1,J+1):
            c_i.append(inner_l[m+j])

        for k in range(1,m+1):
            a_ik.append(inner_l[m+J+k])

        #print("c:",c_i)
        #print("a_ik:",a_ik)

    return n,b_k,c_i,a_ik






#if n == 1 ask



import itertools
n,b_k,c_i,a_ik = read_instance("n_5_m_1_J_2_U_40.txt")
def BF(n,b_k,c_i,a_ik):

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

    
    # print(feas_dec_x)


    #Find Z 
    Z = []

    for x in range(len(feas_dec_x)):

        Z_g = []
        for objective_coef in range(len(c_i)):
            Z_g.append(feas_dec_x[x] @ c_i[objective_coef])
     
        Z.append(Z_g)
    # print(len(Z))
    print(Z)
    #remove duplicated 
    Z_removed_dup = list(set(tuple(z) for z in Z))
    print(len(Z_removed_dup))
    # all x = 0 ask 

    nondominated_Z = Z_removed_dup.copy()


    the_smallest = []

    the_biggest = []
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
    print(nondominated_Z)
    return nondominated_Z