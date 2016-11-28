from fdot import fdot
#from mmul import mmul
import numpy as np
from time import time

y = []
x = np.linspace(10,1000,20)

for i in x:
	a = np.ones((15000,i))
	b = 2*np.ones((i,10))
	ti= time()
	fdot(a,b)#,'\n', time()-ti, '\n'
	tf = time()
	np.dot(a,b)
	tn = time()
	y.append( (tn-tf)/(tf-ti))

print y
#print mmul(a,b)
import matplotlib.pyplot as plt

plt.plot(x,y)
plt.savefig('test1.png')
plt.show()
