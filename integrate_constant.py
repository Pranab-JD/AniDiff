"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Temporal integration
"""

import numpy as np
import os, sys, shutil
from matplotlib import cm
import matplotlib.pyplot as plt
from datetime import datetime

### ------------------------------------------------- ###

### Choose the required initial conditions
# from initial_1 import *       # Ring
# from initial_2 import *       # Periodic Band
# from initial_3 import *       # Analytical
from initial_4 import *       # Gaussian

### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/Constant/")
sys.path.insert(1, "./LeXInt/Python/Constant/Explicit/")
sys.path.insert(1, "./LeXInt/Python/Constant/Implicit/")
sys.path.insert(1, "./LeXInt/Python/Constant/EXPRB/")
# sys.path.insert(1, "./LeXInt/Python/Constant/EPIRK/")

import Eigenvalues
from Explicit import *
from Implicit import *
from EXPRB import *
# from EPIRK import *

### ============================================================================ ###
    
class Integrate(initial_distribution):   
    
    def RHS_function(self, u):
        """
            This function can be used directly for
            explicit and exponential integrators.

        """
        
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
        
        elif integrator == "Crank_Nicolson":
            u_sol, num_rhs_calls = Crank_Nicolson(u, dt, self.A_dif, tol)
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
        print()
        
        ### Create directory
        dt_value = '{:2.0f}'.format(self.N_cfl)
        order = '{:1.0f}'.format(self.spatial_order)
        n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        max_t = '{:1.2f}'.format(self.tmax)
        emax = '{:5.1e}'.format(self.error_tol)
        
        # direc_1 = os.path.expanduser("~/PJD/AniDiff Data Sets/Constant/Ring/" + str(integrator))
        # direc_2 = os.path.expanduser(direc_1 + "/Order_" + str(order) + "/N_" + str(n_x) + "_" + str(n_y) + "/T_" + str(max_t))
        # path = os.path.expanduser(direc_2 + "/dt_" + str(dt_value) + "_CFL/tol " + str(emax) + "/")
        

        # if os.path.exists(path):
        #     shutil.rmtree(path)                                 # remove previous directory with same name
        # os.makedirs(path, 0o777)                                # create directory with access rights
        
        dt_array = []                                           # Array - dt used
        cost_array = []                                         # Array - # of matrix-vector products
        
        time = 0                                                # Time elapsed
        u = self.initial_u().reshape(self.N_x * self.N_y)       # Reshape 2D into 1D
        
        ### Write data for movies
        # file_movie = open(path + "Movie_data.txt", 'w+')
        # file_movie.write(' '.join(map(str, u)) % u + "\n" + "\n")
        
        ############## --------------------- ##############
        
        ### Eigenvalues (Remains constant for linear equations)
        eigen_min_dif = 0.0 
        eigen_max_dif, eigen_imag_dif = Eigenvalues.Gershgorin(self.A_dif)      # Max real, imag eigen value
        
        ### Scaling and shifting factors
        c = 0.5 * (eigen_max_dif + eigen_min_dif)
        Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)
        
        ### Start timer
        tolTime = datetime.now()
        
        ### Time loop ###
        while time < tmax:
            
            ############# --------------------- ##############

            # ### Test plots
            plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = cm.plasma, 
                       extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')
            
            # ax = plt.axes(projection = '3d')
            # ax.grid(False)
            # ax.view_init(elev = 30, azim = 60)
            # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'plasma', edgecolor = 'none')
            
            # plt.colorbar()
            plt.pause(self.dt)
            plt.clf()
            
            ############# --------------------- ##############
            
            if time + self.dt >= tmax:
                self.dt = tmax - time
                
                print("Final time: ", time, " + ", self.dt, " = ", time + self.dt)
                print()
                print("Max. value: ", max(u))
                print("Mean value: ", np.mean(u))
            
            ############## --------------------- ##############
            
            u_sol, num_rhs_calls = self.solve_constant(integrator, u, self.dt, c, Gamma, self.error_tol)
            
            ### Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            
            ### Update variables
            time = time + self.dt
            u = u_sol.copy()
            
            # print("Time elapsed = ", time)
            # print("Mean value: ", np.max(u))
            # print()
            # print()
            
            ### Data for movie
            # file_movie.write(' '.join(map(str, u)) % u + "\n" + "\n")
            
        ### Stop timer
        tol_time = datetime.now() - tolTime
        print(str(tol_time))
        
            
        # ### Write final data to files
        # file_final = open(path + "Final_data.txt", 'w+')
        # np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        # file_final.close()
        
        # # file_movie.close()
        
        # ### Write simulation results to file
        # file_res = open(path + '/Results.txt', 'w+')
        # file_res.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
        # file_res.write("Number of matrix-vector products = %d" % np.sum(cost_array) + "\n" + "\n")
        # file_res.write("dt:" + "\n")
        # file_res.write(' '.join(map(str, dt_array)) % dt_array + "\n" + "\n")
        # file_res.close()
            
### ============================================================================ ###
