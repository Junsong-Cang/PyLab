
LineWidth = 2
FontSize = 15
nx = 100
ny = 200

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

xs = np.logspace(-2, 2, nx)
ys = np.linspace(-5, 5, ny)
z = np.empty((ny, nx))

# Get some fake data
# model = lambda x, y: 2 + np.sin(y) * np.exp(-x**2)

model = lambda x, y: np.exp(-np.log10(x)**2/0.1)*np.exp(-y**2)

for xid in np.arange(0, nx):
    for yid in np.arange(0, ny):
        x_ = xs[xid]
        y_ = ys[yid]
        z[yid, xid] = model(x_, y_)

# Set font
plt.rcParams.update({'font.family':'Times'})
# Use LaTex in axis labels
plt.rcParams['text.usetex'] = True

x, y = np.meshgrid(xs, ys)

fig,ax = plt.subplots()
c=ax.pcolor(x, y, z,cmap='jet',norm = LogNorm(vmin=z.min(), vmax=z.max()))
# Use this for linear colorbar:
# c=ax.pcolor(x, y, z,cmap='jet')

plt.xlabel('$x$',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel('$y$',fontsize=FontSize,fontname='Times New Roman')
plt.xscale('log')

plt.xticks(size=FontSize)
plt.yticks(size=FontSize)

# Colorbar label setting is painstaking
# in many cases it's perhaps easier to just use title+colorbar
# cbar = plt.colorbar(c)
# cbar.ax.tick_params(labelsize=FontSize/1.2)
# ax.text(2.5E11,2.6,'$f_{\\mathrm{bh}}$',rotation=0, size=FontSize)

# fig.colorbar(c, ax=ax,label='$\sigma$')

clb = plt.colorbar(c)
clb.ax.set_title('$z$')

contours = plt.contour(x, y, z, levels=[1e-10, 1e-2], colors=['k','w'])
plt.title('A plot example',fontsize=FontSize)

plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.png',dpi=200)
