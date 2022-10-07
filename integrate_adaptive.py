"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Temporal integration
"""

import numpy as np
import os, sys, shutil
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, kron, identity, diags

from datetime import datetime

### ------------------------------------------------- ###

### Choose the required initial conditions
# from initial_1 import *       # Ring
# from initial_2 import *       # Periodic Band
# from initial_3 import *       # Analytical
from initial_4 import *       # Gaussian


### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/Adaptive/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/Embedded_Explicit/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/EXPRB/")
# sys.path.insert(1, "./LeXInt/Python/Constant/EPIRK/")

import Eigenvalues
from Embedded_explicit import *
from EXPRB import *
# from EPIRK import *

### ============================================================================ ###
    
class Integrate(initial_distribution):   
    
    def RHS_function(self, u):
        
        return self.A_dif.dot(u)
    
    ### ------------------------------------------------- ###
    
    def Laplacian(self):
        
        Laplacian_x = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-2*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1))
        Laplacian_y = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-2*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1))
        
        Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
        Laplacian_y[-1, 0] = 1; Laplacian_x[0, -1] = 1
        
        return kron(identity(self.N_y), Laplacian_x) + kron(Laplacian_y, identity(self.N_x))
        
    def RHS_Laplacian(self, u):
        
        eigen_B = 2
        Laplacian_matrix = self.Laplacian()
        penalty = eigen_B * Laplacian_matrix.dot(u)
        
        return penalty + (self.A_dif.dot(u) - penalty)
    
    ### ------------------------------------------------- ###
    
    def scheme(self, integrator):
        
        if integrator == "Matrix_exp":
                    
            Method_order = 0
            method = real_Leja_exp
            
            return method, Method_order
        
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
                    dt = 0.9 * new_dt                       # Safety factor

                    u_low, u_high, rhs_calls_2 = method(u, dt, self.RHS_function)
                
                    error = np.mean(abs(u_low - u_high))
                
            else:
                
                rhs_calls_2 = 0
                
                new_dt = dt * (tol/error)**(1/(Method_order + 1))
                dt = 0.9 * new_dt                       # Safety factor
                
            # print("Error = ", error)
            
            
        else:
            
            u_low, u_high, rhs_calls_1 = method(u, dt, self.RHS_Laplacian, c, Gamma, tol, 0)
            
            ### Error
            error = np.mean(abs(u_low - u_high))
            
            if error > tol:
                
                new_dt = dt * (tol/error)**(1/(Method_order + 1))
                dt = 0.9 * new_dt                       # Safety factor

                u_low, u_high, rhs_calls_2 = method(u, dt, self.RHS_Laplacian, c, Gamma, tol, 0)
            
                error = np.mean(abs(u_low - u_high))
                
            else:
                
                rhs_calls_2 = 0
                
                new_dt = dt * (tol/error)**(1/(Method_order + 1))
                dt = 0.9 * new_dt                       # Safety factor

            
        return u_high, dt, rhs_calls_1 + rhs_calls_2
        
    ### ------------------------------------------------- ###
        
    def run_code(self, tmax):
        
        ### Choose the integrator
        integrator = "Cash_Karp"
        print("Integrator: ", integrator)
        print()
        
        ### Create directory
        n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        emax = '{:5.1e}'.format(self.error_tol)
        order = '{:1.0f}'.format(self.spatial_order)
        max_t = '{:1.2f}'.format(self.tmax)

        direc_1 = os.path.expanduser("~/PJD/AniDiff Data Sets/Adaptive/Gaussian/" + str(integrator))
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

            ### Test plots
            # plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = cm.plasma, 
            #            extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')
            
            # # ax = plt.axes(projection = '3d')
            # # ax.grid(False)
            # # ax.view_init(elev = 30, azim = 60)
            # # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'plasma', edgecolor = 'none')
            
            # # plt.colorbar()
            # plt.pause(self.dt)
            # plt.clf()
            
            ############# --------------------- ##############
            
            if time + self.dt >= tmax:
                self.dt = tmax - time
                
                print("Final time: ", time, " + ", self.dt, " = ", time + self.dt)
                print()
                print("Max. value: ", max(u))
                print("Mean value: ", np.mean(u))
            
            ############## --------------------- ##############

            u_sol, new_dt, num_rhs_calls = self.solve_adaptive(integrator, u, self.dt, c, Gamma, self.error_tol)
            
            ### Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            time_array.append(time)                             # List of time elapsed
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            eigen_array.append(eigen_max_dif)                   # List of eigenvalues

            # print("Time: ", time)
            # print("dt :", self.dt)
            # print()
            
            ### Update variables
            time = time + self.dt
            u = u_sol.copy()
            self.dt = new_dt
            
        ############# --------------------- ##############
        
        ### Stop timer
        tol_time = datetime.now() - tolTime
        print(str(tol_time))
            
        ### Write final data to files
        file_final = open(path + "Final_data.txt", 'w+')
        np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_final.close()
        
        ### Write simulation results to file
        file_res = open(path + '/Results.txt', 'w+')
        file_res.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
        file_res.write("Number of matrix-vector products = %d" % np.sum(cost_array) + "\n" + "\n")
        file_res.write("dt:" + "\n")
        file_res.write(' '.join(map(str, dt_array)) % dt_array + "\n" + "\n")
        file_res.write("Time:" + "\n")
        file_res.write(' '.join(map(str, time_array)) % time_array + "\n" + "\n")
        file_res.write("Eigenvalues:" + "\n")
        file_res.write(' '.join(map(str, eigen_array)) % eigen_array + "\n" + "\n")
        file_res.close()
            
### ============================================================================ ###
