import autoPyTorch #pip install autoPyTorch
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,1000)
y=3.*x**2.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

cls = autoPyTorch.api.tabular_classification.TabularClassificationTask()
cls.search(x_train, y_train)
predictions = cls.predict(X_test)

plt.clf()
plt.scatter(x_train,y_train,alpha=0.03,label='train')
plt.scatter(x_test,predictions,alpha=0.03,label='predition')
plt.legend()
plt.show()