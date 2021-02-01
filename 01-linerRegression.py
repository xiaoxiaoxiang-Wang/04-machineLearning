import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y = np.array([1,2,2,4,5,6,7,9]).reshape(-1,1)
# plt.plot(x,y,'r',markerfacecolor='blue',marker='o')
# plt.show()

model = LinearRegression() # 构建线性模型
model.fit(x, y) # 自变量在前，因变量在后
predicts = model.predict(x) # 预测值
print(predicts)

# marker 画点
plt.scatter(x,y)
plt.plot(x,predicts,'r',markerfacecolor='blue',marker='o')
plt.show()