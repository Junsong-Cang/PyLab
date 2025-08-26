import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
x = np.linspace(-1, 1, 100)
y = np.sin(x)
fig, ax = plt.subplots()

left, bottom, width, height = [.30, 0.6, 0.2, 0.25]
ax_new = fig.add_axes([left, bottom, width, height])
ax.plot(x, y, color='red')
ax_new.plot(x, y, color='green')

plt.show()
