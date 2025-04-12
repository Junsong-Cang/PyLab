n_tot = 100 # total number of loop

from mpi4py import MPI
import numpy as np
import time
from PyLab import SaySomething

'''
# A mpi version of following for loop, can be run with mpirun
for idx in np.arange(0, n_tot):
    do_something(idx)
'''

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_something(index):
    # Define your processing logic here
    # print(f"Process {rank} is processing index {index}")
    SaySomething(MSG = str(index))
    time.sleep(1)

# Split the range of indices across processes
# 1 - intersecting indexes
# indices = np.arange(rank, n_tot, size)
# 2 - joined indexes
dn = int(n_tot/size)
indices = np.arange(dn*rank, min(dn*(rank+1), n_tot))

print('----', size, '----', rank)
print(indices)

# Each process will work on its own set of indices
for idx in indices:
    do_something(idx)

# Synchronize all processes
comm.Barrier()
