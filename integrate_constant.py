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
from Divided_Difference import *

from ETD import *
from RK import *
from CN import *
from ARK import *
from mu_mode import *
# from kiops import *

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
        
        return self.A_dif.dot(u.reshape(self.N_x * self.N_y))
    
    ### ------------------------------------------------- ###
    
    # def Laplacian(self):
        
    #     Laplacian_x = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-2*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1))
    #     Laplacian_y = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-2*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1))
        
    #     Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
    #     Laplacian_y[0, -1] = 1; Laplacian_y[-1, 0] = 1
        
    #     Laplacian_matrix = kron(identity(self.N_y), Laplacian_x/self.dx**2) + kron(Laplacian_y/self.dy**2, identity(self.N_x))
        
    #     return Laplacian_matrix
    
    # def Linear(self, u):
        
    #     eigen_B = 2
    #     Laplacian_matrix = self.Laplacian()
        
    #     return eigen_B * Laplacian_matrix.dot(u)
    
    ### ------------------------------------------------- ###
    
    def solve_constant(self, integrator, u, dt, c, Gamma, Leja_X, tol, coeffs, exp_Axx, exp_Ayy):
        
        if integrator == "Leja":
            
            u_sol, num_rhs_calls = real_Leja_exp(u, dt, self.RHS_function, c, Gamma, Leja_X, tol, coeffs)
            return u_sol, num_rhs_calls, 0
        
        elif integrator == "Kiops":
            zeroVec = np.zeros(np.shape(u))
            
            u_sol, num_rhs_calls = kiops([1], self.RHS_function_kiops, np.row_stack((u, zeroVec)), tol, task1=True)
            return u_sol, stats
        
        elif integrator == "mu_mode":
            u_sol = mu_mode(u, exp_Axx, exp_Ayy)
            return u_sol, 0
        
        elif integrator == "RK2":
            u_sol, num_rhs_calls = RK2(self.RHS_function, u, dt)
            return u_sol, num_rhs_calls
        
        elif integrator == "RK4":
            u_sol, num_rhs_calls = RK4(self.RHS_function, u, dt)
            return u_sol, num_rhs_calls
        
        elif integrator == "RKF45":
            u_sol, num_rhs_calls = RKF45(self.RHS_function, u, dt)
            return u_sol, num_rhs_calls
        
        elif integrator == "IMEX_Euler":
            u_sol, num_rhs_calls, cost = IMEX_Euler(u, dt, self.A_dif, self.Laplacian, tol)
            return u_sol, num_rhs_calls, cost
        
        elif integrator == "Crank_Nicolson":
            u_sol, num_rhs_calls, cost = Crank_Nicolson(u, dt, self.A_dif, tol)
            return u_sol, num_rhs_calls, cost
        
        elif integrator == "ARK2":
            u_sol, num_rhs_calls, c2, c3 = ARK2(u, dt, self.A_dif, self.Laplacian, tol)
            return u_sol, num_rhs_calls, c2, c3
        
        elif integrator == "ARK4":
            u_sol, num_rhs_calls = ARK4(u, dt, self.A_dif, self.Laplacian, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "ETD1":
            u_sol, num_rhs_calls = ETD1(u, dt, self.Linear, self.RHS_function, c, Gamma, Leja_X, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "ETDRK2":
            u_sol, num_rhs_calls = ETDRK2(u, dt, 2, self.Dif_xx, self.Dif_yy, self.D_xx, self.D_yy, self.dx, self.dy, self.Linear, self.RHS_function, c, Gamma, Leja_X, tol)
            return u_sol, num_rhs_calls
 
        else:
            print("Please choose proper integrator.")
            
    ### ------------------------------------------------- ###
        
    def run_code(self, tmax):
        
        ### Choose the integrator
        integrator = "Leja"
        print("Integrator: ", integrator)
        print()
        
        dt_array = []                                           # Array - dt used
        time_array = []                                         # Array - time elapsed after each time step
        cost_array = []                                         # Array - # of matrix-vector products
        cost_1 = []
        cost_2 = []
        
        time = 0                                                # Time elapsed
        time_steps = 0                                          # Time steps
        u = self.initial_u().reshape(self.N_x * self.N_y)       # Reshape 2D into 1D
        
        
        ### ======================================== ###
        
        ### Write data for movies
        path = os.path.expanduser("~/PrJD/AniDiff Data Sets/Movie/Ring/")
        
        if os.path.exists(path):
            shutil.rmtree(path)                                 # remove previous directory with same name
        os.makedirs(path, 0o777)                                # create directory with access rights
        
        file_movie = open(path + str(time_steps) + ".txt", 'w+')
        np.savetxt(file_movie, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_movie.close()
        
        ### ======================================== ###
        
        ############## --------------------- ##############
        
        ### Read Leja points
        Leja_X = np.loadtxt("Leja_10000.txt")
        Leja_X = Leja_X[0:2000]
        
        ### Eigenvalues (Remains constant for linear equations)
        eigen_min_dif = 0.0
        eigen_max_dif, _ = Gershgorin(self.A_dif)      # Max real, imag eigenvalue
        
        ### Scaling and shifting factors
        c = 0.5 * (eigen_max_dif + eigen_min_dif)
        Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)
        
        ############## --------------------- ##############
        
        ### Start timer
        tolTime = datetime.now()
        
        ### Time loop ###
        while time < tmax:
            
            # filename = 
                    
            file_movie = open(path + str(time_steps) + ".txt", 'w+')
            np.savetxt(file_movie, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
            
            ############# --------------------- ##############

            ### Test plots
            # plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = cm.gist_heat, 
            #            extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')
            
            # # ax = plt.axes(projection = '3d')
            # # ax.grid(False)
            # # ax.view_init(elev = 30, azim = 60)
            # # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'plasma', edgecolor = 'none')
            
            # plt.pause(1)
            # plt.clf()
            
            ############# --------------------- ##############
            
            if time + self.dt >= tmax:
                self.dt = tmax - time
                
                print("Final time: ", time, " + ", self.dt, " = ", time + self.dt)
                print()
                # print("Max. value: ", max(u))
                # print("Mean value: ", np.mean(u))
            
            ############## --------------------- ##############
            
            if (time_steps == 0 or self.dt != dt_array[0]) and integrator == "mu_mode":
                    
                    exp_Axx = linalg.expm(self.Dif_xx/self.dx**2 * self.D_xx * self.dt)
                    exp_Ayy = linalg.expm(self.Dif_yy/self.dy**2 * self.D_yy * self.dt)
                    coeffs = 0
                
            elif (time_steps == 0 or self.dt != dt_array[0]) and integrator == "Leja":
                    
                    ### Matrix exponential (scaled and shifted)
                    matrix_exponential = np.exp((c + Gamma*Leja_X) * self.dt)

                    ### Compute polynomial coefficients
                    coeffs = Divided_Difference(Leja_X, matrix_exponential) 
                    
                    exp_Axx = 0; exp_Ayy = 0
                    
            else:
                exp_Axx = 0; exp_Ayy = 0
                
            ### Call integrator function
            u_sol, num_rhs_calls, c2 = self.solve_constant(integrator, u, self.dt, c, Gamma, Leja_X, self.error_tol, 0, exp_Axx, exp_Ayy)
            
            ### Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            time_array.append(time)                             # List of time elapsed
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            # cost_1.append(c2)
            # cost_2.append(c3) 
            
            ### Update variables
            time = time + self.dt
            time_steps = time_steps + 1
            u = u_sol.copy()
            
            if time_steps%1000 == 0:
                print("Time elapsed: ", time)
            
            # print("Time elapsed = ", time)
            # print("Mean value: ", np.max(u))
            # print()
            # print()
            
        ### ======================================== ###
            
        ## Data for movie
        file_movie = open(path + str(time_steps) + ".txt", 'w+')
        np.savetxt(file_movie, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_movie.close()
        
        ### ======================================== ###

        ############## --------------------- ##############
            
        ### Stop timer
        tol_time = datetime.now() - tolTime
        
        ### Final plots
        # plt.subplot(1, 2, 1)
        # plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = cm.seismic, 
        #                extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')
        # plt.colorbar()
        
        # plt.subplot(1, 2, 2)
        # plt.plot(np.linspace(-1, 1, self.N_x), u.reshape(self.N_x, self.N_y)[int(self.N_x/2), :])
        # plt.plot(np.linspace(-1, 1, self.N_x), 0.1*np.ones(self.N_x), "r--")
        
        # plt.show()
        
        print("# RHS calls  : ", np.sum(cost_array))
        print("Time elapsed : ", str(tol_time))
        print("# time steps : ", time_steps)
        # print("Max value: ", np.max(u))
        
        # # ### Create directory
        # dt_value = '{:0.0f}'.format(self.N_cfl)
        # order = '{:1.0f}'.format(self.spatial_order)
        # n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        # max_t = '{:1.2f}'.format(self.tmax)
        # emax = '{:5.1e}'.format(self.error_tol)
        
        # direc_1 = os.path.expanduser("~/PrJD/AniDiff Data Sets/Constant/Ring/" + str(integrator) + "/Order_2/")
        # direc_2 = os.path.expanduser(direc_1 + "/N_" + str(n_x) + "/T_" + str(max_t))
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
        
        # file_cost = open(path + '/Cost.txt', 'w+')
        # file_cost.write("Total:" + "\n")
        # file_cost.write(' '.join(map(str, cost_array)) % cost_array + "\n" + "\n")
        # file_cost.write("Iters 1:" + "\n")
        # file_cost.write(' '.join(map(str, cost_1)) % cost_1 + "\n" + "\n")
        # # file_cost.write("Iters 2:" + "\n")
        # # file_cost.write(' '.join(map(str, cost_2)) % cost_2 + "\n" + "\n")
        # file_cost.close()
        
### ============================================================================ ###