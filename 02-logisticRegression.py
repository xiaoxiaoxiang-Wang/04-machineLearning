import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression

x = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y = np.array([0,0,0,1,0,1,1,1]).reshape(-1)
x1 = np.array([1,2,3,5.1,4.2,6,7,8]).reshape(-1,1)

lr = LogisticRegression(solver='lbfgs').fit(x,y)
predicts = lr.predict_proba(x)[:,1]

plt.scatter(x,y)
plt.plot(x,predicts,'r',markerfacecolor='blue',marker='o')
plt.show()


print(predicts)