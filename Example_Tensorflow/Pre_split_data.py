import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

x=np.linspace(0,10,1000)
y=3.*x**2.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


plt.clf()
plt.scatter(x_train,y_train,alpha=0.03,label='train')
plt.scatter(x_test,y_test,alpha=0.03,label='test')
plt.legend()
plt.show()