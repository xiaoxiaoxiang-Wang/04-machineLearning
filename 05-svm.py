import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.linspace(0, 10, 100)
y = np.sin(x*10)+1
y1 = np.linspace(0, 10, 100)
for i in range(1,len(y1)):
    y1[i]=0.01*y1[i-1]+0.99*y[i]
y2 = y1-y
# plt.scatter(x,y)
plt.xticks(np.arange(0,10,0.5))
plt.yticks(np.arange(-2,2,0.2))
plt.plot(x,y,'r',markerfacecolor='blue',marker='o')
plt.plot(x,y1,'r',markerfacecolor='red',marker='o')
# plt.plot(x,y2,'r',markerfacecolor='yellow',marker='o')
plt.show()