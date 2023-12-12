"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

"""

from integrate_constant import *
from datetime import datetime

startTime = datetime.now()

### ============================================================================ ###

###! Parameters
tmax_list = [0.75]
N_list = [2**8]

cfl_list = [1000]

# tol_list = [1e-4, 1e-5, 1e-6, 1e-7]
# tol_list = [1e-8, 1e-9, 1e-10]

tol_list = [1e-8]

for N in N_list:
    for tmax in tmax_list:
        for n_cfl in cfl_list:
            for tol in tol_list:

                ### Object initialization
                run = Integrate(N, N, tmax, n_cfl, tol)

                def main():
                    run.run_code(tmax)
                    
                if __name__ == "__main__":
                    main()


### ============================================================================ ###

print()
print("==============================================")
print()
print('Total Time Elapsed = ', datetime.now() - startTime)
print()
print("Wall-clock time: ", datetime.now())
print()