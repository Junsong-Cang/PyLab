import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.random.rand(100)
y = np.random.rand(100)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axes array for easy indexing
axs = axs.flatten()

for i in range(4):
    axs[i].scatter(x, y, color='blue', s=5)
    axs[i].set_title(f"Panel {i+1}")
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")

plt.tight_layout()
plt.show()
