#from: https://github.com/dynamicslab/pysindy
#documentation: https://pysindy.readthedocs.io/en/latest/

#install with pip
#pip install pysindy

import pysindy

import matplotlib.pyplot as plt

import numpy as np
import pysindy as ps

t = np.linspace(0, 1, 100)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y

plt.clf()
plt.plot(x,label="x")
plt.plot(y,label="y")
plt.yscale('log')
plt.legend()
plt.show()


model = ps.SINDy(feature_names=["x", "y"])
model.fit(X, t=t)

model.print()