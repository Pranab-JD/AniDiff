"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

Description:
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# from integrate_adaptive import *
from integrate_constant import *

from datetime import datetime

startTime = datetime.now()

### ============================================================================ ###

### Parameters
spatial_order = 2
tmax_list = [4.0]
# N_list = [2**6, 2**7]
N_list = [2**6]

cfl_list = [100]
# cfl_list = [10]

tol_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
# tol_list = [1e-7]

for N in N_list:
    for tmax in tmax_list:
        for n_cfl in cfl_list:
            for tol in tol_list:

                ### Object initialization
                run = Integrate(N, N, spatial_order, tmax, n_cfl, tol)

                def main():
                    run.run_code(tmax)
                    
                if __name__ == "__main__":
                    main()


### Test
# run = Integrate(2**7, 2**7, 2, 0.2, 500, 1e-8)

# def main():
#     run.run_code(0.2)
    
# if __name__ == "__main__":
#     main()


### ============================================================================ ###

print()
print("==============================================")
print()
print('Total Time Elapsed = ', datetime.now() - startTime)
print()