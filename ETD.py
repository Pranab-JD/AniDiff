"""
Created on Fri Apr 12 11:46:41 2022

@author: Pranab JD
"""

import sys
import numpy as np

### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/")

from Phi_functions import *
from real_Leja_exp import *
from real_Leja_phi_nl import *

from mu_mode import *

from datetime import datetime

def ETD1(u, dt, eigen_B, A_xx, A_yy, D_xx, D_yy, dx, dy, Linear, RHS, c, Gamma, Leja_X, tol):
    
    u_linear = mu_mode(u, dt, eigen_B, A_xx, A_yy, D_xx, D_yy, dx, dy)
    
    F_u = (RHS(u) - Linear(u))*dt
    u_nonlin, rhs_phi, conv = real_Leja_phi(u, dt, Linear, F_u, c, Gamma, Leja_X, phi_1, tol)
    
    u_etd1 = u_linear + u_nonlin
    
    return u_etd1, rhs_phi + 2


def ETDRK2(u, dt, eigen_B, A_xx, A_yy, D_xx, D_yy, dx, dy, Linear, RHS, c, Gamma, Leja_X, tol):
    
    u_linear = mu_mode(u, dt, eigen_B, A_xx, A_yy, D_xx, D_yy, dx, dy)
    
    F_u = (RHS(u) - Linear(u)) * dt
    u_nonlin_1, rhs_phi_1, conv = real_Leja_phi(u, dt, Linear, F_u, c, Gamma, Leja_X, phi_1, tol)
    
    a = u_linear + u_nonlin_1
    
    F_a = (RHS(a) - Linear(a)) * dt
    u_nonlin_2, rhs_phi_2, conv = real_Leja_phi(u, dt, Linear, F_a - F_u, c, Gamma, Leja_X, phi_2, tol)
    
    u_etdrk2 = a + u_nonlin_2
    
    return u_etdrk2, rhs_phi_1 + rhs_phi_2 + 4