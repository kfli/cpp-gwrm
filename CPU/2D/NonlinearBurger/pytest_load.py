import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from sys import argv

data = np.loadtxt(fname = 'array_data.txt', delimiter = ', ')
print(data)

X = []
Y = []
Z = []
for i in range(len(data)):
    for j in range(3):
        if j == 0:
            X.append(data[i][j])
        elif j == 1:
            Y.append(data[i][j])
        else:
            Z.append(data[i][j])

#pp.plot(X, Y, Z)
#pp.xlabel('X')
#pp.ylabel('T')
#pp.title('U')
#pp.legend()
#pp.show()
fig = pp.figure()
# Subplot 1D
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
#fig.colorbar(surf, shrink=0.5, aspect=5)

pp.show()
