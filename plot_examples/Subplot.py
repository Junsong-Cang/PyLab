# Get subplot

LineWidth = 2
FontSize = 18

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x**2)

# Set axis font and LaTex
plt.rcParams.update({
    'text.usetex': True,
    'font.family':'Times',
    'text.latex.preamble': r'\usepackage{newtxtext,newtxmath}'})
# Set plot number
fig, axs = plt.subplots(1, 2, sharex = False, sharey = False)
# adjust figure width
fig.set_size_inches(8, 4)

# if plots are more than one line use this:
# axs[0, 0], axs[0, 1]
idx = 0
axs[idx].grid(True, which='major', linewidth = 0.2) # Show grid, if u wanna show both minor and major grid, use which='both'
axs[idx].plot(x, y1, 'k', linewidth = LineWidth, label='sin')
axs[idx].plot(x, y2, 'r', linewidth = LineWidth, label='cos')
axs[idx].legend(fontsize=FontSize, loc = 'upper left')
axs[idx].set_title('First Plot',fontsize=FontSize)
axs[idx].set_xlabel('$x$',fontsize=FontSize,fontname='Times New Roman')
axs[idx].set_ylabel('$y$',fontsize=FontSize,fontname='Times New Roman')
axs[idx].tick_params(axis='both', which='both', labelsize = FontSize)
# Set axis limits
axs[idx].set_xlim(-3.2, 3.2)
axs[idx].set_ylim(-1.2, 1.2)
# Can set x and y scales between linear (default) and log
axs[idx].set_yscale('linear')
# Can set tick numbers like this:
# axs[idx, Tb_idx].set_yticks(np.array([-1500, -1000, -500, 0]))

idx = 1
axs[idx].plot(x, y3, 'k', linewidth = LineWidth, label='exp')
axs[idx].legend(fontsize=FontSize, loc = 'upper left')
axs[idx].set_title('Second Plot',fontsize=FontSize)
axs[idx].set_xlabel('$x$',fontsize=FontSize,fontname='Times New Roman')
axs[idx].set_ylabel('$y$',fontsize=FontSize,fontname='Times New Roman')
axs[idx].set_xlim(-3.2, 3.2)
axs[idx].set_ylim(0, 1.2)

plt.xticks(size=FontSize)
plt.yticks(size=FontSize)
# You can also use: axs[idx].set_xticks

plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')
