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
from initial_1 import *       # Ring
# from initial_2 import *       # Band
# from initial_3 import *       # Analytical
# from initial_4 import *       # Gaussian


### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/")
sys.path.insert(1, "./LeXInt/Python/Constant/")

from Eigenvalues import *
from real_Leja_exp import *
from ETD import *

from CN import *
from ARK import *
from mu_mode import *
from kiops import *

### ============================================================================ ###
    
class Integrate(initial_distribution):   
    
    def RHS_function(self, u):
        """
            This function can be used directly for
            explicit and exponential integrators.

        """
        
        return self.A_dif.dot(u)
    
    def RHS_function_kiops(self, u):
        """
            This function can be used directly for
            explicit and exponential integrators.

        """
        
        # print(np.shape(u))
        
        return self.A_dif.dot(u.reshape(self.N_x * self.N_y))
    
    ### ------------------------------------------------- ###
    
    def Laplacian(self):
        
        Laplacian_x = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-2*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1))
        Laplacian_y = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-2*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1))
        
        Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
        Laplacian_y[-1, 0] = 1; Laplacian_y[0, -1] = 1
        
        return kron(identity(self.N_y), Laplacian_x) + kron(Laplacian_y, identity(self.N_x))
    
    def Linear(self, u):
        
        eigen_B = -5
        
        return eigen_B * self.Laplacian().dot(u)
    
    ### ------------------------------------------------- ###
    
    def solve_constant(self, integrator, u, dt, c, Gamma, Leja_X, tol):
        
        if integrator == "Leja":
            u_sol, num_rhs_calls = real_Leja_exp(u, dt, self.RHS_function, c, Gamma, Leja_X, tol)
            return u_sol, num_rhs_calls
        
        if integrator == "Kiops":
            zeroVec = np.zeros(np.shape(u))
            
            u_sol, num_rhs_calls = kiops([1], self.RHS_function_kiops, np.row_stack((u, zeroVec)), tol, task1=True)
            return u_sol, stats
        
        elif integrator == "mu_mode":
            u_sol = mu_mode(u, dt, self.Dif_xx, self.Dif_yy, \
                                   self.D_xx, self.D_yy, self.dx, self.dy)
            return u_sol, 0
        
        elif integrator == "Crank_Nicolson":
            u_sol, num_rhs_calls = Crank_Nicolson(u, dt, self.A_dif, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "ARK2":
            u_sol, num_rhs_calls = ARK2(u, dt, self.A_dif, self.Laplacian, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "ETD1":
            u_sol, num_rhs_calls = ETD1(u, dt, self.Linear, self.RHS_function, c, Gamma, Leja_X, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "ETDRK2":
            u_sol, num_rhs_calls = ETDRK2(u, dt, self.Linear, self.RHS_function, c, Gamma, Leja_X, tol)
            return u_sol, num_rhs_calls
        
        # elif integrator == "RosEu":
        #     u_sol, num_rhs_calls = Rosenbrock_Euler(u, dt, self.RHS_function, c, Gamma, Leja_X, tol, 0)
        #     return u_sol, num_rhs_calls
        
        # elif integrator == "EXPRB32":
        #     u_sol, num_rhs_calls = EXPRB32(u, dt, self.RHS_Laplacian, c, Gamma, Leja_X, tol, 0)
        #     return u_sol, num_rhs_calls
        
        # elif integrator == "EPIRK4s3A":
        #     u_sol, num_rhs_calls = EPIRK4s3A(u, dt, self.RHS_Laplacian, c, Gamma, Leja_X, tol, 0)
        #     return u_sol, num_rhs_calls
 
        else:
            print("Please choose proper integrator.")
            
    ### ------------------------------------------------- ###
        
    def run_code(self, tmax):
        
        ### Choose the integrator
        integrator = "mu_mode"
        print("Integrator: ", integrator)
        print()
        
        ### Read Leja points
        Leja_X = np.loadtxt("Leja_10000.txt")
        Leja_X = Leja_X[0:1000]
        
        dt_array = []                                           # Array - dt used
        time_array = []                                         # Array - time elapsed after each time step
        cost_array = []                                         # Array - # of matrix-vector products
        
        time = 0                                                # Time elapsed
        time_steps = 0                                          # Time steps
        u = self.initial_u().reshape(self.N_x * self.N_y)       # Reshape 2D into 1D
        
        ### Write data for movies
        # file_movie = open(path + "Movie_data.txt", 'w+')
        # file_movie.write(' '.join(map(str, u)) % u + "\n" + "\n")
        
        ############## --------------------- ##############
        
        ### Eigenvalues (Remains constant for linear equations)
        eigen_min_dif = 0.0
        eigen_max_dif, eigen_imag_dif = Gershgorin(self.A_dif)      # Max real, imag eigenvalue
        
        ### Scaling and shifting factors
        c = 0.5 * (eigen_max_dif + eigen_min_dif)
        Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)
        
        ### Start timer
        tolTime = datetime.now()
        
        ### Time loop ###
        while time < tmax:
            
            ############# --------------------- ##############

            ### Test plots
            plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = cm.gist_heat, 
                       extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')
            
            # ax = plt.axes(projection = '3d')
            # ax.grid(False)
            # ax.view_init(elev = 30, azim = 60)
            # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'plasma', edgecolor = 'none')
            
            plt.pause(self.dt)
            plt.clf()
            
            ############# --------------------- ##############
            
            if time + self.dt >= tmax:
                self.dt = tmax - time
                
                print("Final time: ", time, " + ", self.dt, " = ", time + self.dt)
                print()
                # print("Max. value: ", max(u))
                # print("Mean value: ", np.mean(u))
            
            ############## --------------------- ##############
            
            u_sol, num_rhs_calls = self.solve_constant(integrator, u, self.dt, c, Gamma, Leja_X, self.error_tol)
            
            ### Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            time_array.append(time)                             # List of time elapsed
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            
            ### Update variables
            time = time + self.dt
            time_steps = time_steps + 1
            u = u_sol.copy()
            
            if time_steps%10000 == 0:
                print("Time elapsed: ", time)
            
            # print("Time elapsed = ", time)
            # print("Mean value: ", np.max(u))
            # print()
            # print()
            
            ### Data for movie
            # file_movie.write(' '.join(map(str, u)) % u + "\n" + "\n")

        ############## --------------------- ##############
            
        ### Stop timer
        tol_time = datetime.now() - tolTime
        
        ### Final plots
        # plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = cm.plasma, 
        #                extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')
        # plt.colorbar()
        # plt.show()
        
        print("# RHS calls: ", np.sum(cost_array))
        print("Time elapsed: ", str(tol_time))
        print("# time steps: ", time_steps)
        
        ### Create directory
        # dt_value = '{:2.0f}'.format(self.N_cfl)
        # order = '{:1.0f}'.format(self.spatial_order)
        # n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        # max_t = '{:1.2f}'.format(self.tmax)
        # emax = '{:5.1e}'.format(self.error_tol)
        
        # direc_1 = os.path.expanduser("~/PrJD/AniDiff Data Sets/Constant/Band/" + str(integrator))
        # direc_2 = os.path.expanduser(direc_1 + "/Order_" + str(order) + "/N_" + str(n_x) + "/T_" + str(max_t))
        # path = os.path.expanduser(direc_2 + "/dt_" + str(dt_value) + "_CFL/tol " + str(emax) + "/")
        
        # if os.path.exists(path):
        #     shutil.rmtree(path)                                 # remove previous directory with same name
        # os.makedirs(path, 0o777)                                # create directory with access rights
            
        # ### Write final data to file
        # file_final = open(path + "Final_data.txt", 'w+')
        # np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        # file_final.close()
        
        # ### Write simulation results to file
        # file_res = open(path + '/Results.txt', 'w+')
        # file_res.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
        # file_res.write("Number of matrix-vector products = %d" % np.sum(cost_array) + "\n" + "\n")
        # file_res.write("dt:" + "\n")
        # file_res.write(' '.join(map(str, dt_array)) % dt_array + "\n" + "\n")
        # file_res.write("Time:" + "\n")
        # file_res.write(' '.join(map(str, time_array)) % time_array + "\n" + "\n")
        # file_res.close()
        
        # # file_movie.close()
        
### ============================================================================ ###