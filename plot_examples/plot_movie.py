LineWidth = 2
FontSize = 18

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
import matplotlib.animation as ani

plt.rcParams.update({'font.family':'Times'})
# Use LaTex in axis labels
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)

# fig = plt.figure()
l, = plt.plot([], [], 'k', linewidth=LineWidth, label = 'sin')
plt.xlabel('$x$',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel('$y$',fontsize=FontSize,fontname='Times New Roman')
plt.xticks(size=FontSize)
plt.yticks(size=FontSize)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title('A movie plot example',fontsize=FontSize)
plt.text(3, 0.5, "Text", size=FontSize, rotation=0,color='k')
plt.legend(fontsize=FontSize,loc = 'lower left')
plt.tight_layout()

def func(x):
    return np.sin(x)*3

'''
xlist = np.linspace(-5, 5, 100)
ylist = func(xlist)
l.set_data(xlist, ylist)
plt.show()
'''

metadata = dict(title = 'Movie', artist='codinglikemad')
writer = PillowWriter(fps = 15, metadata=metadata)

xlist = []
ylist = []

# 100 is dpi
with writer.saving(fig = fig, outfile = "sinWave.gif", dpi = 400):
    for xval in np.linspace(-5, 5, 100):
        xlist.append(xval)
        ylist.append(func(xval))
        # ylist.append(xval)
        l.set_data(xlist, ylist)
        writer.grab_frame()
