import numpy as num 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
from source_region import GR

import matplotlib.pyplot as plt
from scipy import stats

import scipy.stats as st


class GutenbergRichter(stats.rv_continuous):
    def _pdf(self, x, aval, bval):
        return 10**(aval-bval*x)

test = GutenbergRichter(a=0, b=8, name='test', shapes="aval, bval")
print test
print 'sadfasdf'
plt.hist(test.rvs(3,0.5,  size=1000))
plt.show()
data = GutenbergRichterDiscrete(0,1)
print data
plt.plot(data[0], data[1], 'o')
plt.show()
xk = num.arange(7)
pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1)
#h = plt.plot(data[0], custm.pmf(data[0]))
#plt.figure()
custm = stats.rv_discrete(name='custm', values=(xk, pk))
h = plt.plot(xk, custm.pmf(xk))
plt.figure()
data =[]
for i in range(1000):
    data.append( custm.rvs())
print data
plt.hist(data)

plt.show()


a = [[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]]
b = num.array(a).T
c = num.array([[1,2],[3,4]])
b = b.reshape(7,2)
b = b.T
i = 0 
for v in num.nditer(b, op_flags=['readwrite'],flags=['external_loop'], order='C'):
    b[...] = num.dot(c,v)

    i +=1

