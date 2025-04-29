
import numpy as np
import math
import matplotlib.pyplot as plt

#Variables

t_orb = 47520 #min
ecl = 0.01
P_ecl = 450 #W
buss_V=28 #V

k=1.2
el_path_ef=0.8 #Debatable
DoD = 0.3
cell_cap = 58 #Ah
cell_dis_t = 7 #h
C_V = 3.5

def Enegry_req_spacecraft():
    return P_ecl*t_orb*ecl*60
def actual_capacity():
    C_act= (P_ecl*t_orb*ecl/60)/(buss_V*DoD*el_path_ef)
    ecl_t=t_orb*ecl/60
    C_nom = ((C_act**k)*(cell_dis_t**(k-1))/(ecl_t**(k-1)))**(1/k)
    return C_act,C_nom

print(Enegry_req_spacecraft()/1000)
print(actual_capacity()[1])
print(round(actual_capacity()[1]/cell_cap)*buss_V/C_V)
print(round(actual_capacity()[1]/cell_cap))