'''
FileName = '/home/dm/gaolq/cjs/21cmFAST_cache/MCMC_cache_10/LC_0.002644_-1.258330_-1.220660_4.381547E+01_0.205800.h5'

from p21c_tools import *
import py21cmfast as p21c
import time

t1 = time.time()
LightCone = p21c.LightCone.read(FileName)
d = PowerSpectra(LightCone)
t2 = time.time()
print(' time = ', t2 - t1)

print(d)
'''

import numpy as np
a = np.array([0.1, 1, 3, np.nan, np.nan, 32])
idx = np.isnan(a)
idx = [3, 4]
idx = np.arange(0,2)

na = np.delete(a, idx)

print(a)
print(na)

print(idx)
