import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import linprog

np.set_printoptions(precision = 3)
angles = pd.read_csv('a4q2_headingangles.csv',header = None)
T = 100
vmin = 0.01
xf = 3
yf = 4
x0 = 0
y0 = 0
heading_angles = angles.to_numpy()
heading_angles = heading_angles.reshape(1,100)
heading_angles[0,T-1] = 0
#print(heading_angles)
A = np.vstack((np.cos(heading_angles),np.sin(heading_angles)))
#A = np.round_(A,3)
#print(A)

c = -1*np.ones((1,T))
 
b = np.array([xf-x0,yf-y0])
b = b.reshape(2,1)
res = linprog(c, A_eq=A, b_eq=b, bounds= (vmin,None))
v = res.x
x = np.zeros(T+1)
y = np.zeros(T+1)
x[0] = x0
y[0] = y0

fig = plt.figure()
plt.xlabel('X axis', fontsize=18)
plt.ylabel('Y axis', fontsize=16)
plt.scatter(x0,y0,color = 'black',label = 'initial point')
plt.scatter(xf,yf,color = 'brown',label = 'final point')

for i in range(0,T):
    x[i+1] = x[i] + v[i]*math.cos(heading_angles[0,i])
    y[i+1] = y[i] + v[i]*math.sin(heading_angles[0,i])
    #plt.scatter(x[i+1],y[i+1],color = 'black')
plt.plot(x,y,color = 'r',label = 'vmin = 0.01')
plt.legend()
plt.show()

print(res.x)