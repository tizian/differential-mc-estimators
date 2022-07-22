import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

N = 256
vals = np.ones((N, 4))
vals[:, 0] = 0
vals[:, 1] = 0
vals[:, 2] = 0

vals[:N//2, 0] = np.linspace(0, 1, N//2)[::-1]
vals[N//2:, 1] = np.linspace(0, 1, N//2)

cmap_diff = ListedColormap(vals)
plt.register_cmap(name='diff', cmap=cmap_diff)
