# The start of every file should contain the libraries you import.
import numpy as np
import subprocess
import re


# Below function is used to get function value,gradient and hessian at a given point
# Input: point at which function value needs to be obtained
# Output: returns function value, gradient and Hessian after passing it through oracle
def readPoint(SR, currPoint):
    input = str(SR) + ",[" + ','.join(str(elem) for elem in currPoint) + "]"
    someVar = subprocess.run(["Q3_oracle_2.exe", input], stdout=subprocess.PIPE).stdout.decode('utf-8')
    func_val = re.search(r'(.*)', someVar).group(1)
    func_val = float(func_val)
    gr = re.search(r'\[(.*?)\]',someVar).group(1)
    hess = re.findall(r'\[\[(.*)\]\]', someVar, flags = re.DOTALL)
    Arr1 = gr.split(",")
    X = str(hess)
    X = re.sub('[\'\[\]]', '', X)
    X = X.replace('\\r\\n','')
    Arr2 = X.split(",")
    FloatArr1 = [float(i) for i in Arr1]
    FloatArr2 = [float(i) for i in Arr2]
    gradient = np.array(FloatArr1,dtype = np.float64)
    Hessian  = np.array(FloatArr2,dtype = np.float64)
    Hessian  = np.reshape(Hessian,(4,4))
    return func_val,gradient,Hessian

# Below function performs back tracking line search to get a good step size
# Input: point at which the search has to be performed
# Output: returns point after Back tracking line search , step size, function value 

def backTrackingLineSearch(x0):
    i = 0
    func_val_x0,gradient,Hessian = readPoint(19598,x0)
    descent_dir   = -1*gradient
    currPoint = x0
    curr_func_val = np.random.random(100)
    t = 1
    while i<30:
        currPoint = x0 + t*descent_dir
        curr_func_val = readPoint(19598,currPoint)[0]
        if curr_func_val <= func_val_x0 + 0.5*t*np.dot(descent_dir,gradient):
            break
        elif np.linalg.norm(t*descent_dir) <= pow(10,-7):
            break
        else:
            t = 0.5*t
            i = i+1
    
    return currPoint,t,curr_func_val

# Below function performs the Newton Method
# Input: SR Number and the initial point (In this case [5,-3,-5,3])
# Output: Number of iterations,function value when Newton Method converges
def NewtonMethod(SR,x0):
    iters = 0
    currPoint = x0
    func_val,gradient,Hessian = readPoint(19598,x0)
    step_size = 1
    while iters < 50 and np.linalg.norm(gradient) > pow(10,-10):
        descent_dir = -np.dot(np.linalg.inv(Hessian),gradient)
        currPoint = currPoint + step_size*descent_dir
        if iters == 0 or iters== 4 or iters ==9 or iters==19:
            print("x%d=%s" % (iters+1, currPoint))
        func_val,gradient,Hessian = readPoint(SR, currPoint)
        iters += 1
    return iters,func_val


# Below function performs the actual gradient descent
# Input: SR Number and the initial point (In this case [5,-3,-5,3])
# Output: Number of iterations,function value when gradient descent converges
def gradientDescent(SR, x0):
    iters = 0
    currPoint = x0
    func_val,gradient,Hessian = readPoint(19598,x0)
    while iters < 50 and np.linalg.norm(gradient) > pow(10,-10):
        step_size = backTrackingLineSearch(currPoint)[1]
        currPoint = currPoint - step_size*gradient
        if iters == 0 or iters== 4 or iters ==9 or iters==19:
            print("for x%d function value =%s" % (iters+1, readPoint(SR, currPoint)[0]))
        func_val,gradient,Hessian  = readPoint(SR, currPoint)
        iters += 1
    return iters,func_val

if __name__ == '__main__':
    
    np.set_printoptions(precision=3)
    iters,f_val = gradientDescent(19598,np.array([5,-3,-5,3]))
    print('For Gradient Descent iters = %d function value = %s '%(iters,f_val))
