import numpy as num 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
from source_region import GR

import matplotlib.pyplot as plt
from scipy import stats

import scipy.stats as st

class my_pdf(st.rv_continuous):
    def _pdf(self,x):
        return x*x/10.0

my_cv = my_pdf(a=0.0,b=1.0,name='my_pdf')
print my_cv.rvs(size=10)


a = [[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]]
b = num.array(a).Ty
c = num.array([[1,2y],[3,4]])
b = b.reshape(7,2)
print b
b = b.T
i = 0 
for v in num.nditer(b, op_flags=['readwrite'],flags=['external_loop'], order='C'):
    print v, i
    print num.dot(c,v)
    b[...] = num.dot(c,v)


    i +=1

print b
