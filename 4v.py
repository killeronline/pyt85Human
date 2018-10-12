import matplotlib.pyplot as plt
import numpy as np


def GenY(x,h):
    return np.ones(np.shape(x))*h


x = np.random.randint(0,100,100)

#plt.plot(x, GenY(x,1) , ',',label='raw', color='blue' )


m = np.array([x.mean()])

#plt.plot(m, GenY(m,2) , '.',label='mean {0}'.format(m), color='green' )


std = np.array([x.std()])

#plt.plot(std, GenY(std,3) , 'd',label='std {0}'.format(std), color='red' )


n = (x - m)/std

plt.plot(n, GenY(n,4) , '+',label='norm', color='grey' )


m2 = np.array([n.mean()])

plt.plot(m2, GenY(m2,5) , '.',label='mean of norm {0}'.format(m2), color='green' )


std2 = np.array([n.std()])

plt.plot(std2, GenY(std2,6) , 'd',label='std of norm {0}'.format(std2), color='red' )




plt.legend()
plt.show()
