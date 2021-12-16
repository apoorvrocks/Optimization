import numpy as np
import math
from sympy import *
from scipy.optimize import minimize,dual_annealing


# takes x as input
# returns function value, first derivative,second derivative 
def readPoint(x0):
    x = Symbol('x')
    y = 0.5*log(x**2+1)
    grad = y.diff(x)
    Hessian = grad.diff(x)
    return y.evalf(subs={x:x0}),grad.evalf(subs={x:x0}),Hessian.evalf(subs={x:x0})

# implementation of Newton Method
def NewtonMethod(x0):
    iters = 0
    currPoint = x0
    func_val_x0,gradient,Hessian = readPoint(x0)
    step_size = 1
    while abs(gradient) > 0.05:
        descent_dir = -gradient/Hessian ## 1 dimensional case 
        currPoint = currPoint + step_size*descent_dir
        func_val,gradient,Hessian = readPoint(currPoint)
        iters += 1
    return iters, currPoint

## return the absolute value of third derivative

def abs_third_derivative(x):
    return abs(8.0*x**3/(x**2 + 1)**3 - 6.0*x/(x**2 + 1)**2)




if __name__ == '__main__':
    # part 5b to find maximum value of third derivative in order to find Lipschitz constant of Hessian
    M = 0
    for i in np.arange(0,1,0.01):
        M = max(M,abs_third_derivative(i))   
    print('Lipschitz constant for Hessian = %s'%(M))
    print('a = %f for newton region' %(2/(3*M))) ## newton region = 2/3*(a/M) where a = 1 as shown in pdf 
    
    # part 5c
    iters, currPoint = NewtonMethod(0.1)
    print('x = 0.1 , iters = %d ,currpoint = %s'%(iters, currPoint))
    iters, currPoint = NewtonMethod(0.6)
    print('x = 0.6 , iters = %d ,currpoint = %s'%(iters, currPoint))
    iters, currPoint = NewtonMethod(-0.5)
    print('x = -0.5 , iters = %d ,currpoint = %s'%(iters, currPoint))
    iters, currPoint = NewtonMethod(1.2)
    print('x = 1.2 , iters = %d ,currpoint = %s'%(iters, currPoint))