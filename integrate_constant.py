"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Temporal integration
"""

import numpy as np
import os, sys, shutil
from matplotlib import cm
import matplotlib.pyplot as plt

### Choose the required initial conditions
# from initial_1 import *
from initial_2 import *
# from initial_3 import *

### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/Adaptive/")
sys.path.insert(1, "./LeXInt/Python/Constant/Explicit/")
sys.path.insert(1, "./LeXInt/Python/Constant/EPIRK/")
sys.path.insert(1, "./LeXInt/Python/Constant/EXPRB/")

import Eigenvalues
from Explicit import *
from EXPRB import *
from EPIRK import *

### ============================================================================ ###
    
class Integrate(initial_distribution):   
    
    def RHS_function(self, u):
        
        return self.A_dif.dot(u)
    
    ### ------------------------------------------------- ###
    
    def solve_constant(self, integrator, u, dt, c, Gamma, tol):
        
        if integrator == "Matrix_exp":
            u_sol, num_rhs_calls = real_Leja_exp(u, dt, self.RHS_function, c, Gamma, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "RK2":
            u_sol, num_rhs_calls = RK2(u, dt, self.RHS_function)
            return u_sol, num_rhs_calls
            
        elif integrator == "RK4":
            u_sol, num_rhs_calls = RK4(u, dt, self.RHS_function)
            return u_sol, num_rhs_calls
            
        elif integrator == "RKF45":
            u_sol, num_rhs_calls = RKF45(u, dt, self.RHS_function)
            return u_sol, num_rhs_calls
        
        elif integrator == "DOPRI54":
            u_sol, num_rhs_calls = DOPRI54(u, dt, self.RHS_function)
            return u_sol, num_rhs_calls
        
        elif integrator == "Cash_Karp":
            u_sol, num_rhs_calls = Cash_Karp(u, dt, self.RHS_function)
            return u_sol, num_rhs_calls

        elif integrator == "Rosenbrock_Euler":
            u_sol, num_rhs_calls = Rosenbrock_Euler(u, dt, self.RHS_function, c, Gamma, tol, 0)
            return u_sol, num_rhs_calls
            
        elif integrator == "EXPRB42":
            u_sol, num_rhs_calls = EXPRB42(u, dt, self.RHS_function, c, Gamma, tol, 0)
            return u_sol, num_rhs_calls
 
        else:
            print("Please choose proper integrator.")
            
    ### ------------------------------------------------- ###
        
    def run_code(self, tmax):
        
        ### Choose the integrator
        integrator = "Matrix_exp"
        print("Integrator: ", integrator)
        
        ### Create directory
        dt_value = '{:5.1e}'.format(self.dt)
        direc_1 = os.path.expanduser("~/PJD/AniDiff Data Sets/Test Order/" + str(integrator))
        path = os.path.expanduser(direc_1 + "/dt " + str(dt_value) + "/")

        if os.path.exists(path):
            shutil.rmtree(path)                                 # remove previous directory with same name
        os.makedirs(path, 0o777)                                # create directory with access rights
        
        dt_array = []                                           # Array - dt used
        cost_array = []                                         # Array - # of matrix-vector products
        
        time = 0                                                # Time elapsed
        u = self.initial_u().reshape(self.N_x * self.N_y)       # Reshape 2D into 1D
        tol = 1e-5                                              # Tolerance for polynomial interpolation
        
        ############## --------------------- ##############
        
        ### Time loop ###
        while time < tmax:
            
            if time + self.dt >= tmax:
                self.dt = tmax - time
                
                print("Final time: ", time, " + ", self.dt, " = ", time + self.dt)
                print()
                print("Max. value: ", max(u))
                print("Mean value: ", np.mean(u))
            
            ############## --------------------- ##############
            
            ### Eigenvalues
            eigen_min_dif = 0.0 
            eigen_max_dif, eigen_imag_dif = Eigenvalues.Gershgorin(self.A_dif)      # Max real, imag eigen value
            
            ### Scaling and shifting factors
            c = 0.5 * (eigen_max_dif + eigen_min_dif)
            Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)

            u_sol, num_rhs_calls = self.solve_constant(integrator, u, self.dt, c, Gamma, tol)
            
            ### Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            
            ### Update variables
            time = time + self.dt
            u = u_sol.copy()
            
            print("Time elapsed = ", time)
            
            ############# --------------------- ##############

            ### Test plots
            # plt.imshow(u.reshape(self.N_y, self.N_x), origin = 'lower', cmap = cm.gist_heat, extent = [0, 1, 0, 1], aspect = 'equal')
            
            # # ax = plt.axes(projection = '3d')
            # # ax.grid(False)
            # # ax.view_init(elev = 30, azim = 120)
            # # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'plasma', edgecolor = 'none')
            
            
            # plt.pause(self.dt/10)
            # plt.clf()
            
            ############# --------------------- ##############
            
        ### Write final data to files
        file_final = open(path + "Final_data.txt", 'w+')
        np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_final.close()
        
        ### Write simulation results to file
        file_res = open(path + '/Results.txt', 'w+')
        file_res.write("Number of matrix-vector products = %d" % np.sum(cost_array) + "\n" + "\n")
        file_res.write("dt:" + "\n")
        file_res.write(' '.join(map(str, dt_array)) % dt_array + "\n" + "\n")
        file_res.close()
        
        plt.imshow(u.reshape(self.N_y, self.N_x), origin = 'lower', cmap = cm.gist_heat, extent = [0, 1, 0, 1], aspect = 'equal')
        plt.colorbar()
        plt.savefig("../AniDiff Data Sets/anidiff_5.eps")
            
### ============================================================================ ###