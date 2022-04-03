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

startTime = datetime.now()

### ------------------------------------------------- ###

### Choose the required initial conditions
# from initial_1 import *       # Ring
from initial_2 import *         # Periodic Band
# from initial_3 import *       # Analytical
# from initial_4 import *       # Ring
# from initial_5 import *       # Gaussian Pulse

### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/Adaptive/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/Embedded_Explicit/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/EPIRK/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/EXPRB/")

import Eigenvalues
from Embedded_explicit import *
from EXPRB import *
from EPIRK import *

### ============================================================================ ###
    
class Integrate(initial_distribution):   
    
    def RHS_function(self, u):
        
        return self.A_dif.dot(u)
    
    ### ------------------------------------------------- ###
    
    def scheme(self, integrator):
        
        if integrator == "Matrix_exp":
                    
            Method_order = 0
            method = Leja.real_Leja_exp
        
        elif integrator == "EXPRB32":
                
            Method_order = 2
            method = EXPRB32
            
            return method, Method_order
            
        elif integrator == "EXPRB43":
            
            Method_order = 3
            method = EXPRB43
            
            return method, Method_order
            
        elif integrator == "EXPRB54s4":
            
            Method_order = 4
            method = EXPRB54s4
            
            return method, Method_order

        elif integrator == "EXPRB54s5":

            Method_order = 4
            method = EXPRB54s5
            
            return method, Method_order
            
        elif integrator == "EPIRK5P1":

            Method_order = 4
            method = EPIRK5P1
            
            return method, Method_order
                
        elif integrator == "RKF45":

            Method_order = 4
            method = RKF45
            
            return method, Method_order
        
        elif integrator == "DOPRI54":
    
            Method_order = 4
            method = DOPRI54
            
            return method, Method_order
            
        elif integrator == "Cash_Karp":
    
            Method_order = 4
            method = Cash_Karp
            
            return method, Method_order
                
        else:
            print("Please choose proper integrator.")
            
    ### ------------------------------------------------- ###
    
    def solve_adaptive(self, integrator, u, dt, c, Gamma, tol):
        
        method, Method_order = self.scheme(integrator)
            
        if integrator == "RKF45" or integrator == "DOPRI54" or integrator == "Cash_Karp":
            
            u_low, u_high, rhs_calls_1 = method(u, dt, self.RHS_function)
            
            ### Error
            error = np.mean(abs(u_low - u_high))
            
            if error > tol:
                
                while error > tol:
                
                    new_dt = dt * (tol/error)**(1/(Method_order + 1))
                    dt = 0.8 * new_dt                       # Safety factor

                    u_low, u_high, rhs_calls_2 = method(u, dt, self.RHS_function, c, Gamma, tol, 0)
                
                    error = np.mean(abs(u_low - u_high))
                
            else:
                
                rhs_calls_2 = 0
                
                new_dt = dt * (tol/error)**(1/(Method_order + 1))
                dt = 0.8 * new_dt                       # Safety factor
                
            # print("Error = ", error)
            
            
        else:
            
            u_low, u_high, rhs_calls_1 = method(u, dt, self.RHS_function, c, Gamma, tol, 0)
            
            ### Error
            error = np.mean(abs(u_low - u_high))
            
            if error > tol:
                
                new_dt = dt * (tol/error)**(1/(Method_order + 1))
                dt = 0.8 * new_dt                       # Safety factor

                u_low, u_high, rhs_calls_2 = method(u, dt, self.RHS_function, c, Gamma, tol, 0)
            
                error = np.mean(abs(u_low - u_high))
                
            else:
                
                rhs_calls_2 = 0
                
                new_dt = dt * (tol/error)**(1/(Method_order + 1))
                dt = 0.8 * new_dt                       # Safety factor

            
        return u_high, dt, rhs_calls_1 + rhs_calls_2
        
    ### ------------------------------------------------- ###
        
    def run_code(self, tmax):
        
        ### Choose the integrator
        integrator = "DOPRI54"
        print("Integrator: ", integrator)
        print()
        
        ### Create directory
        n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        emax = '{:5.1e}'.format(self.error_tol)
        order = '{:1.0f}'.format(self.spatial_order)
        max_t = '{:1.2f}'.format(self.tmax)

        direc_1 = os.path.expanduser("~/PJD/AniDiff Data Sets/Adaptive/Band_Constant/" + str(integrator))
        direc_2 = os.path.expanduser(direc_1 + "/Order_" + str(order) + "/N_" + str(n_x) + "_" + str(n_y) + "/T_" + str(max_t))
        path = os.path.expanduser(direc_2 + "/tol " + str(emax) + "/")

        if os.path.exists(path):
            shutil.rmtree(path)                                 # remove previous directory with same name
        os.makedirs(path, 0o777)                                # create directory with access rights
        
        dt_array = []                                           # Array - dt used
        time_array = []                                         # Array - time elapsed after each time step
        cost_array = []                                         # Array - # of matrix-vector products
        eigen_array = []                                        # Array - eigenvalues
        
        time = 0                                                # Time elapsed
        u = self.initial_u().reshape(self.N_x * self.N_y)       # Reshape 2D into 1D
        
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

            u_sol, new_dt, num_rhs_calls = self.solve_adaptive(integrator, u, self.dt, c, Gamma, self.error_tol)
            
            ### Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            time_array.append(time)                             # List of time elapsed
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            eigen_array.append(eigen_max_dif)                   # List of eigenvalues

            # print("dt :", self.dt)
            # print("Max eigenvalue :", eigen_max_dif)
            # print("Cost :", num_rhs_calls)
            
            ### Update variables
            time = time + self.dt
            u = u_sol.copy()
            self.dt = new_dt
            
            ############# --------------------- ##############

            ### Test plots
            # plt.imshow(u.reshape(self.N_y, self.N_x), origin = 'lower', cmap = cm.gist_heat, extent = [0, 1, 0, 1], aspect = 'equal')
            
            # ax = plt.axes(projection = '3d')
            # ax.grid(False)
            # ax.view_init(elev = 30, azim = 120)
            # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'plasma', edgecolor = 'none')
            
            # plt.pause(self.dt/10)
            # plt.clf()
            
        ############# --------------------- ##############
        
        ### Stop timer
        simulation_time = datetime.now() - startTime
        print(str(simulation_time))
            
        ### Write final data to files
        file_final = open(path + "Final_data.txt", 'w+')
        np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_final.close()
        
        ### Write simulation results to file
        file_res = open(path + '/Results.txt', 'w+')
        file_res.write("Time elapsed (secs): %s" % str(simulation_time) + "\n" + "\n")
        file_res.write("Number of matrix-vector products = %d" % np.sum(cost_array) + "\n" + "\n")
        file_res.write("dt:" + "\n")
        file_res.write(' '.join(map(str, dt_array)) % dt_array + "\n" + "\n")
        file_res.write("Time:" + "\n")
        file_res.write(' '.join(map(str, time_array)) % time_array + "\n" + "\n")
        file_res.write("Eigenvalues:" + "\n")
        file_res.write(' '.join(map(str, eigen_array)) % eigen_array + "\n" + "\n")
        file_res.close()
        
        # plt.imshow(u.reshape(self.N_y, self.N_x), origin = 'lower', cmap = cm.gist_heat, extent = [0, 1, 0, 1], aspect = 'equal')
        # plt.colorbar()
        # plt.savefig("../AniDiff Data Sets/anidiff.eps")
            
### ============================================================================ ###
