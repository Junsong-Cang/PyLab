
file = 'data/tmp.h5'

import numpy as np
import h5py
# some mock data
x = np.linspace(0,2*np.pi,50)
y = np.sin(x)

# ----save data----
f = h5py.File(file, 'w')
f.create_dataset('dataset_x', data = x)
f.create_dataset('dataset_y', data = y)
f.close()

# ----load data----
# x1 and y1 are np arrays
f = h5py.File(file, 'r')
x1 = f['dataset_x'][:]
y1 = f['dataset_y'][:]
redshift = f.attrs['redshift']# Read attribute
f.close()
print(x1)

# h5py tutorial ended, you can do a small plot now
import matplotlib.pyplot as plt
plt.plot(x1,y1)
plt.show()
