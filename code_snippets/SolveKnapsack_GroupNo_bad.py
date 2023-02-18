'''
imports
'''

# Not allowed
from my_pkg.my_mod import my_custom_fn

# Allowed, Helper
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



def SolveKnapsack(filename, method):  
  '''
  Logic
  '''

# Not allowed
time.sleep(1000)
# Other random commands