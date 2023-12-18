"""
Created on Wed Dec  1 15:37:38 2021

@author: Pranab JD

"""

from integrate_constant import *
from datetime import datetime

startTime = datetime.now()

### ============================================================================ ###

###! Parameters

### Spiral 1 = (X + 4Y, -X), T = 0.6, 1.0, 3.0
### Spiral 2 = (X + Y,  -X), T = 0.6, 1.0, 3.0

tmax_list = [3.0]

cfl_list = [1000]

tol_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# tol_list = [1e-12]

N = 2**8

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

print('Total Time Elapsed = ', datetime.now() - startTime)
print()
print("Wall-clock time: ", datetime.now())
print()
print("====================================================================")
print()