import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from pysindy import SINDy


def lorenz(z,t):
	return [10*(z[1] - z[0]),\
            z[0]*(28 - z[2]) - z[1],\
            z[0]*z[1] - 8/3*z[2],\
            ]
'''
lorenz = lambda z,t : [10*(z[1] - z[0]),\
                       z[0]*(28 - z[2]) - z[1],\
                       z[0]*z[1] - 8/3*z[2],\
                       ]
'''

t = np.arange(0,2,.002)
x = solve_ivp(lorenz, (0,2), [-8,8,27],max_step=t[1]-t[0])

#x solves dz / dt = lorenz(t, z)

model = SINDy()
model.fit(x, t=t[1]-t[0])
model.print()

plot=True
if plot:
	plt.clf()
	plt.plot(x,t,label="x")
	plt.legend()
	plt.show()