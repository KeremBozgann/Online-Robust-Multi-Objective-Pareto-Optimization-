from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
median= 0
norm.ppf(0.9, loc= median , scale= 1)

t_bar= 0.2
t= np.linspace(0, t_bar, 1000)
def deviation(t, median):
    quantile= np.zeros(t.shape)
    for i, _t in enumerate(t):
        quantile[i]= max(norm.ppf(1 / 2 + _t) - median, median - norm.ppf(1 / 2 - _t))
    return quantile

dev= deviation(t, median)
fig, ax= plt.subplots(1)
ax.scatter(t,dev)
plt.show()

k= (dev[-1] - dev[0])/(t[-1] -t[0])
R=lambda t: t * k

dev_bound= R(t)
fig, ax= plt.subplots(1)
ax.scatter(t,dev_bound)
plt.show()
np.all(dev_bound >= dev)

