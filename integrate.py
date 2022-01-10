"""
Created on Wed Dec  3 15:46:29 2021

@author: Pranab JD

Description: Temporal integration
"""

import os, sys, shutil

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

### Choose the required initial conditions
# from initial_1 import *
from initial_2 import *
# from initial_3 import *

### Import LeXInt
sys.path.insert(1, "./LeXInt/Python/Adaptive/Embedded_Explicit/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/EPIRK/")
sys.path.insert(1, "./LeXInt/Python/Adaptive/EXPRB/")

from Embedded_explicit import *
from EXPRB import *
from EPIRK import *

### ============================================================================ ###
    
class Integrate(initial_distribution):   
    
    def RHS_function(self, u):
        
        return self.A_dif.dot(u)
    
    
    def solve(self, integrator, u, dt):
        
        if integrator == "RKF45":
            u_low, u_high, num_rhs_calls = RKF45(u, dt, self.RHS_function)
            
        return u_low, u_high, num_rhs_calls
        
        
    def run_code(self, tmax):
        
        ### Create directory
        n_x = '{:2.0f}'.format(self.N_x); n_y = '{:2.0f}'.format(self.N_y)
        emax = '{:5.1e}'.format(self.error_tol)
        dt_value = '{:5.1e}'.format(self.dt)
        direc_1 = os.path.expanduser("~/PJD/AniDiff Data Sets/Test Order/RKF45/5/")
        path = os.path.expanduser(direc_1 + "/dt " + str(dt_value) + "/")

        if os.path.exists(path):
            shutil.rmtree(path)                     # remove previous directory with same name
        os.makedirs(path, 0o777)                    # create directory with access rights
        
        t = 0
        u = self.initial_u().reshape(self.N_x * self.N_y)       # Reshape 2D into 1D
        
        ############## --------------------- ##############
        
        ### Time loop
        while t < tmax:
            
            if t + self.dt >= tmax:
                self.dt = tmax - t
                
                print("Final time: ", t, " + ", self.dt, " = ", t + self.dt)
                print()
                print("Max. value: ", max(u))
                print("Mean value: ", np.mean(u))
            
            ############## --------------------- ##############

            u_sol_low, u_sol_high, num_rhs_calls = self.solve("RKF45", u, self.dt)
            
            ### Update variables
            t = t + self.dt
            u = u_sol_high.copy()
            
            # print("Time: ", t)
            
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
            
        ### Write final data to files
        file_final = open(path + "Final_data.txt", 'w+')
        np.savetxt(file_final, u.reshape(self.N_y, self.N_x), fmt = '%.25f')
        file_final.close()
        
        plt.imshow(u.reshape(self.N_y, self.N_x), origin = 'lower', cmap = cm.gist_heat, extent = [0, 1, 0, 1], aspect = 'equal')
        plt.colorbar()
        plt.savefig("../AniDiff Data Sets/1.eps")
        
        print("Diffusion coeffs: ", self.D_xx, self.D_xy, self.D_yx, self.D_yy)
            
### ============================================================================ ###