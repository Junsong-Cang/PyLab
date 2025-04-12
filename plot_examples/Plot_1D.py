# Get 1D plot

LineWidth = 2
FontSize = 18

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,2*np.pi,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# Set axis font and LaTex
plt.rcParams.update({
    'text.usetex': True,
    'font.family':'Times',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}'})
fig, ax = plt.subplots()
ax.grid(True, which='both', linewidth = 0.3)  # `which='both'` enables major and minor grids

# adjust figure width
fig.set_size_inches(10, 8)

plt.plot(x, y1, '-k', linewidth=LineWidth, label = 'sin')

# To set log scale:
# plt.yscale('log')
# or : plt.loglog
# or you can use:
# plt.plot(x,y1,color = 'k',linestyle='-',linewidth=LineWidth,label = 'sin')
plt.plot(x, y2, '-b', linewidth=LineWidth, label = 'cos')
plt.xlabel('$x$',fontsize=FontSize,fontname='Times New Roman')
plt.ylabel('$y$',fontsize=FontSize,fontname='Times New Roman')

# Set ticks and size
plt.xticks(np.arange(-1,8,1),size=FontSize)
plt.yticks(size=FontSize)

# adjust figure size (slim or long)
plt.title('A plot example',fontsize=FontSize)

# Add text
plt.text(3, 0.5, "Text", size=FontSize, rotation=0,color='k')

# Set legend
plt.legend(fontsize=FontSize,loc = 'lower left')
# or you can alse use
# plt.legend(['sin','cos'],fontsize=FontSize,loc = 'lower left')

# axis ticks

# Set axsi limits
plt.xlim([-1,7])
plt.ylim([-1.2,1.2])
plt.tight_layout()

# Save plot to a file
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')
# plt.show()
