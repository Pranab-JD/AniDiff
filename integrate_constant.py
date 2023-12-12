"""
Created on Wed Dec 3 15:46:29 2021

@author: Pranab JD

Description: Temporal integration
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, kron, identity, diags

from datetime import datetime

### =============================================== ###

### Choose the required initial conditions
# from initial_1 import *       # Ring
# from initial_2 import *       # Band
from initial_3 import *       # Square
# from initial_4 import *       # Gaussian

###! Import LeXInt
sys.path.insert(1, "../LeXInt/Python/")
sys.path.append("..")

from Jacobian import Jacobian
from linear_phi import linear_phi
from Eigenvalues import Gershgorin
from real_Leja_linear_exp import real_Leja_linear_exp

###! Other solvers
from ETD import *
from RK import *
from CN import *
from ARK import *
from mu_mode import *

### ============================================================================ ###
### ============================================================================ ###
    
class Integrate(initial_distribution):
    
    ###! Time-independent sources
    def TIS(self):
        
        S = 0.1 + 30*np.exp(-((self.X + 0.6)**2 + (self.Y - 0.75)**2)/0.025) \
                + 40*np.exp(-((self.X - 0.75)**2 + (self.Y + 0.8)**2)/0.03)
        
        return S.reshape(self.N_x * self.N_y)
    
    ###! Time-dependent sources
    def TDS(self, *args, c):
        
        t = args[0]; dt = args[1]
            
        S1 = 100 * np.exp(-((self.X - 0.25)**2 + (self.Y - 0.25)**2)/0.015) * np.exp(-10 * abs(0.1 - (t + c*dt)))
        S2 =  70 * np.exp(-((self.X - 0.4)**2  + (self.Y + 0.4)**2)/0.025)  * np.exp(-5 * abs(0.3 - (t + c*dt)))
        S3 =  50 * np.exp(-((self.X + 0.6)**2  + (self.Y - 0.6)**2)/0.015)  * np.exp(-7.5 * abs(0.6 - (t + c*dt)))
        
        return (S1 + S2 + S3).reshape(self.N_x * self.N_y)
    
    ### =============================================== ###
    
    ###! RHS function (MAIN)
    def RHS_function(self, u):
        
        return (self.A_dif).dot(u)
    
    ###! RHS function (time-independent sources)
    def RHS_phi(self, u):

        ###? Time-independent source
        S = 0.1 + 30*np.exp(-((self.X + 0.6)**2 + (self.Y - 0.75)**2)/0.025) \
                + 40*np.exp(-((self.X - 0.75)**2 + (self.Y + 0.8)**2)/0.03)

        return self.RHS_function(u) + S.reshape(self.N_x * self.N_y)
    
    ###! RHS function (explicit solvers)
    def RHS_explicit(self, u, *args):
        
        ###? Time-dependent source
        Source = self.TDS(*args, c = 0)
        
        return self.RHS_function(u) + Source
    
    ###! RHS function (exponential quadrature solvers)
    def RHS_Euler(self, u, *args):
               
        ###? Time-dependent source
        Source = self.TDS(*args, c = 0)
        
        return self.RHS_function(u) + Source
        
    def RHS_midpoint(self, u, *args):

        ###? Time-dependent source
        Source = self.TDS(*args, c = 1/2)
    
        return self.RHS_function(u) + Source
    
    def RHS_trapezoidal(self, u, *args):
        
        ###? Time-dependent source
        Source_n  = self.TDS(*args, c = 0)
        Source_n1 = self.TDS(*args, c = 1)
    
        return self.RHS_function(u) + Source_n, Source_n1 - Source_n
    
    def RHS_Gauss(self, u, *args):
    
        c_1 = 0.5 * (1 - 1/3**0.5)
        c_2 = 0.5 * (1 + 1/3**0.5)
        
        ###? Time-dependent sources
        Source_1 = ((3**0.5 + 1)/2) * self.TDS(*args, c = c_1) + ((1 - 3**0.5)/2) * self.TDS(*args, c = c_2)
        Source_2 = self.TDS(*args, c = c_2) - self.TDS(*args, c = c_1)

        return self.RHS_function(u) + Source_1, Source_2
    
    ### =============================================== ###
    
    ###! Functions for penalisation method
    def Laplacian(self):
        
        Laplacian_x = lil_matrix(diags(np.ones(self.N_x - 1), -1) + diags(-2*np.ones(self.N_x), 0) + diags(np.ones(self.N_x - 1), 1))
        Laplacian_y = lil_matrix(diags(np.ones(self.N_y - 1), -1) + diags(-2*np.ones(self.N_y), 0) + diags(np.ones(self.N_y - 1), 1))
        
        Laplacian_x[0, -1] = 1; Laplacian_x[-1, 0] = 1
        Laplacian_y[0, -1] = 1; Laplacian_y[-1, 0] = 1
        
        Laplacian_matrix = kron(identity(self.N_y), Laplacian_x/self.dx**2) + kron(Laplacian_y/self.dy**2, identity(self.N_x))
        
        return Laplacian_matrix
    
    def Linear(self, u):
        
        eigen_B = 2
        Laplacian_matrix = self.Laplacian()
        
        return eigen_B * Laplacian_matrix.dot(u)

    ### =============================================== ###
    
    def solve_constant(self, integrator, u, substeps, c, Gamma, Leja_X, tol, exp_Axx, exp_Ayy, *args):
        
        ###? Differentiate t and dt (for simplicity)
        t = args[0]; dt = args[1]

        ###? Vector of zeros
        zero_vec = np.zeros(np.shape(u))

        ###? Jacobian function for augmented matrix
        Jac_vec_dt = lambda z: dt * Jacobian(self.RHS_function, u, z, self.RHS_function(u))
        
        if integrator == "Leja_exp":
            u_sol, num_rhs_calls, substeps = real_Leja_linear_exp(u, dt, substeps, Jac_vec_dt, c, Gamma, Leja_X, tol)
            return u_sol, num_rhs_calls+1, substeps

        elif integrator == "Exponential_TIS":
            u_flux, num_rhs_calls, substeps = linear_phi([zero_vec, self.RHS_TIS(u)*dt], dt, substeps, Jac_vec_dt, c, Gamma, Leja_X, tol)
            return u + u_flux, num_rhs_calls+1, substeps
        
        elif integrator == "Exponential_Euler":
            u_flux, num_rhs_calls, substeps = linear_phi([zero_vec, self.RHS_Euler(u, *args)*dt], dt, substeps, Jac_vec_dt, c, Gamma, Leja_X, tol)
            return u + u_flux, num_rhs_calls+1, substeps
        
        elif integrator == "Exponential_midpoint":
            u_flux, num_rhs_calls, substeps = linear_phi([zero_vec, self.RHS_midpoint(u, *args)*dt], dt, substeps, Jac_vec_dt, c, Gamma, Leja_X, tol)
            return u + u_flux, num_rhs_calls+1, substeps
        
        elif integrator == "Exponential_trapezoidal":
            interp_vectors = self.RHS_trapezoidal(u, *args)
            u_flux, num_rhs_calls, substeps = linear_phi([zero_vec, interp_vectors[0]*dt, interp_vectors[1]*dt], dt, substeps, Jac_vec_dt, c, Gamma, Leja_X, tol)
            return u + u_flux, num_rhs_calls+1, substeps
        
        elif integrator == "Exponential_Gauss":
            interp_vectors = self.RHS_Gauss(u, *args)
            u_flux, num_rhs_calls, substeps = linear_phi([zero_vec, interp_vectors[0]*dt, 3**0.5*interp_vectors[1]*dt], dt, substeps, Jac_vec_dt, c, Gamma, Leja_X, tol)
            return u + u_flux, num_rhs_calls+1, substeps

        elif integrator == "mu_mode":
            u_sol = mu_mode(u, exp_Axx, exp_Ayy)
            return u_sol, 0
        
        elif integrator == "RK2":
            u_sol, num_rhs_calls = RK2(self.RHS_function, u, dt)
            return u_sol, num_rhs_calls
        
        elif integrator == "RK4":
            u_sol, num_rhs_calls = RK4(self.RHS_explicit, u, dt, *args)
            return u_sol, num_rhs_calls
        
        elif integrator == "RKF45":
            u_sol, num_rhs_calls = RKF45(self.RHS_function, u, dt)
            return u_sol, num_rhs_calls
        
        elif integrator == "IMEX_Euler":
            u_sol, num_rhs_calls = IMEX_Euler(u, dt, self.A_dif, self.Laplacian, tol)
            return u_sol, num_rhs_calls
        
        elif integrator == "Crank_Nicolson":
            u_sol, num_rhs_calls = Crank_Nicolson(u, dt, self.A_dif + self.A_adv, tol, self.TDS(*args, c = 0), self.TDS(args[0] + dt, args[1], c = 0))
            return u_sol, num_rhs_calls
        
        elif integrator == "ARK2":
            u_sol, num_rhs_calls, c2, c3 = ARK2(u, dt, self.A_dif, self.Laplacian, tol)
            return u_sol, num_rhs_calls
        
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
            
    ### ============================================================================ ###
    ### ============================================================================ ###

    def run_code(self, tmax):
        
        ###! Choose the integrator
        integrator = "Leja_exp"
        print("Integrator: ", integrator)
        print()
        
        dt_array = []                                           # Array - dt used
        time_array = []                                         # Array - time elapsed after each time step
        cost_array = []                                         # Array - # of matrix-vector products
        
        time = 0                                                # Time elapsed
        time_steps = 0                                          # Time steps
        
        ###! Initial condition
        u = self.initial_u().reshape(self.N_x * self.N_y)     # Reshape 2D into 1D
        # u = np.ones((self.N_x * self.N_y)) * 1e-2
        
        ############## --------------------- ##############
        
        ###! Write data to files for movie
        # path = os.path.expanduser("~/PrJD/AniDiff Data Sets/Movie/Ring/")
        
        # if os.path.exists(path):
        #     shutil.rmtree(path)                                 # remove previous directory with same name
        # os.makedirs(path, 0o777)                                # create directory with access rights
        
        # file_movie = open(path + str(time_steps) + ".txt", 'w+')
        # np.savetxt(file_movie, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        # file_movie.close()
        
        ############## --------------------- ##############
        
        ###? Read Leja points
        Leja_X = np.loadtxt("Leja_10000.txt")
        Leja_X = Leja_X[0:1000]
        
        ###? Eigenvalues (Remains constant for linear equations)
        eigen_min_dif = 0.0
        eigen_max_dif, _ = Gershgorin(self.A_dif)      # Max real, imag eigenvalue
        
        ###? Scaling and shifting factors
        c = 0.5 * (eigen_max_dif + eigen_min_dif)
        Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)

        ###? Initial guess for substeps
        substeps = 2
        
        ############## --------------------- ##############
        
        ### Start timer
        tolTime = datetime.now()
        
        ###!! Time loop !!###
        while time < tmax:

            ############# --------------------- ##############

            ###? Test plots
            plt.imshow(u.reshape(self.N_x, self.N_y), origin = 'lower', cmap = plt.cm.seismic, 
                       extent = [self.xmin, self.xmax, self.ymin, self.ymax], aspect = 'equal')

            # for ii in range(self.N_x):
            #     for jj in range(self.N_y):
            #         if u.reshape(self.N_x, self.N_y)[ii, jj] < 0:
            #             plt.plot(self.X[ii, ii], self.Y[jj, jj], "r.")

            # # ax = plt.axes(projection = '3d')
            # # ax.grid(False)
            # # ax.view_init(elev = 30, azim = 60)
            # # ax.plot_surface(self.X, self.Y, u.reshape(self.N_y, self.N_x), cmap = 'seismic', edgecolor = 'none')

            plt.colorbar()
            plt.pause(0.5)
            plt.clf()

            ############# --------------------- ##############
            
            ###! Final time step
            if time + self.dt >= tmax:
                self.dt = tmax - time
                
                # print("Final time: ", time, " + ", self.dt, " = ", time + self.dt)
                # print()
                # print("Max. value: ", max(u))
                # print("Min value: ", np.min(u))
            
            ############## --------------------- ##############
            
            ###! Compute the matrix exponentials for the mu-mode interator
            ###! only at the first and last time steps
            
            if (time_steps == 0 or self.dt != dt_array[0]) and integrator == "mu_mode":
                    
                    exp_Axx = linalg.expm(self.Dif_xx/self.dx**2 * self.D_xx * self.dt)
                    exp_Ayy = linalg.expm(self.Dif_yy/self.dy**2 * self.D_yy * self.dt)
            else:
                exp_Axx = 0; exp_Ayy = 0
                
            ############## --------------------- ##############
                
            ###? Call integrator function
            u_sol, num_rhs_calls, substeps = self.solve_constant(integrator, u, substeps, c, Gamma, Leja_X, self.error_tol, exp_Axx, exp_Ayy, time, self.dt)
            
            ###? Append data to arrays
            dt_array.append(self.dt)                            # List of dt at each time step
            time_array.append(time)                             # List of time elapsed
            cost_array.append(num_rhs_calls)                    # List of matrix-vector products
            
            ###? Update variables
            time = time + self.dt
            time_steps = time_steps + 1
            u = u_sol.copy()

            min_u = np.unravel_index(u.argmin(), u.shape)
            print("Min value: ", np.min(u))
            
            if time_steps%100 == 0:
                print()
                print("Time steps: ", time_steps)
                print("Time elapsed: ", time)
                print("RHS calls: ", np.sum(cost_array))
                print("Max value: ", np.max(u))
                print("Min value: ", np.min(u))
                print("Substeps used: ", substeps)
                print()
                           
            ############## --------------------- ##############
                
            ###! Write data to files for movie
            # file_movie = open(path + str(time_steps) + ".txt", 'w+')
            # np.savetxt(file_movie, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
            # file_movie.close()

        ############## --------------------- ##############
            
        ### Stop timer
        tol_time = datetime.now() - tolTime

        print("\n----------------------------")
        print("RHS calls    : ", np.sum(cost_array))
        print("Time elapsed : ", str(tol_time))
        print("Time steps   : ", time_steps)
        print("Minimum value: ", np.min(u))
        print("----------------------------\n")
        
        ### ============================================================================ ###
        
        ###! Create directories
        # dt_value = '{:0.0f}'.format(self.N_cfl)
        # n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        # max_t = '{:1.2f}'.format(self.tmax)
        # emax = '{:5.1e}'.format(self.error_tol)
        
        # direc_1 = os.path.expanduser("../AniDiff_Source_Data/Constant/Spiral_2_DAS/" + str(integrator))
        # direc_2 = os.path.expanduser(direc_1 + "/N_" + str(n_x) + "/T_" + str(max_t))
        # path = os.path.expanduser(direc_2 + "/dt_" + str(dt_value) + "_CFL/tol " + str(emax) + "/")
        
        # if os.path.exists(path):
        #     shutil.rmtree(path)                                 # remove previous directory with same name
        # os.makedirs(path, 0o777)                                # create directory with access rights
            
        # ###! Write final data to file
        # file_final = open(path + "Final_data.txt", 'w+')
        # np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        # file_final.close()
        
        # ###! Write simulation results to file
        # file_res = open(path + '/Results.txt', 'w+')
        # file_res.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
        # file_res.write("Number of matrix-vector products = %d" % np.sum(cost_array) + "\n" + "\n")
        # file_res.write("dt:" + "\n")
        # file_res.write(' '.join(map(str, dt_array)) % dt_array + "\n" + "\n")
        # file_res.write("Time:" + "\n")
        # file_res.write(' '.join(map(str, time_array)) % time_array + "\n" + "\n")
        # file_res.close()
        
### ============================================================================ ###
### ============================================================================ ###