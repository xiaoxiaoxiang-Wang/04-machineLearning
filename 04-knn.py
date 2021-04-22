import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
# y = np.random.rand(100)
#
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2, 0.4)
z = neigh.fit(samples)
print(z)
# plt.plot(x,y,'ro')
# plt.show()

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 2]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) # doctest: +ELLIPSIS
KNeighborsClassifier(...)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
