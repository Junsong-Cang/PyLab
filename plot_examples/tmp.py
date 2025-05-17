import matplotlib.pyplot as plt
import numpy as np

n = 10000
x = np.random.rand(n)
y = np.random.rand(n)
z = ((x - 0.2)**2 + y**2)**0.5
# z = np.random.rand(n)  # This is your "likelihood"

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)

# Create scatter plots
for id1 in [0, 1]:
    for id2 in [0, 1]:
        sc = axes[id1, id2].scatter(x, y, c=z, cmap='viridis', s = 0.2)
        cbar = fig.colorbar(sc, ax=axes[id1, id2], shrink = 0.9, pad = 0.01)
        

# Add a single shared colorbar

plt.tight_layout()
plt.savefig('/Users/cangtao/Desktop/tmp.pdf')
