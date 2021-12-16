import numpy as np
import math
from sympy import *

# Below function is used to get function value,gradient and hessian at a given point
# Input: point at which function value needs to be obtained
# Output: returns function value, gradient and Hessian 
def readPoint(x0):
    x = Symbol('x')
    y = 0.5*log(x**2+1)
    grad = y.diff(x)
    Hessian = grad.diff(x)
    return y.evalf(subs={x:x0}),grad.evalf(subs={x:x0}),Hessian.evalf(subs={x:x0})

# Below function performs back tracking line search to get a good step size
# Input: point at which the search has to be performed
# Output: returns point after Back tracking line search , step size, function value 

def backTrackingLineSearch(x0):
    i = 0
    func_val_x0,gradient,Hessian = readPoint(x0)
    descent_dir   = -1*gradient
    currPoint = x0
    curr_func_val = np.random.random(100)
    t = 1
    while i<30: ## max iterations = 30
        currPoint = x0 + t*descent_dir
        curr_func_val = readPoint(currPoint)[0]
        if curr_func_val <= func_val_x0 + 0.5*t*descent_dir*gradient: ## alpha = 0.5
            break
        elif abs(t*descent_dir) < 0.01: ## epsilon =0.01
            break 
        else:
            t = 0.1*t  ## beta = 0.1
            i = i+1
    
    return currPoint,t,curr_func_val


# Below function performs the actual gradient descent
# Input:  initial point 
# Output: Number of iterations,curr point when gradient descent converges
def gradientDescent(x0):
    iters = 0
    currPoint = x0
    func_val_x0,gradient,Hessian = readPoint(x0)
    while abs(gradient) > 0.05:
       
        step_size = backTrackingLineSearch(currPoint)[1]
        currPoint = currPoint - step_size*gradient
        gradient  = readPoint(currPoint)[1]
        iters += 1
        
    return iters,currPoint

if __name__ == '__main__':
    iters,currPoint = gradientDescent(0.1)
    print('x = 0.1 , iters = %d ,currpoint = %s'%(iters, currPoint))
    iters,currPoint = gradientDescent(0.6)
    print('x = 0.6 , iters = %d ,currpoint = %s'%(iters, currPoint))
    iters,currPoint = gradientDescent(-0.5)
    print('x = -0.5 , iters = %d ,currpoint = %s'%(iters, currPoint))
    iters,currPoint = gradientDescent(1.2)
    print('x = 1.2 , iters = %d ,currpoint = %s'%(iters, currPoint))