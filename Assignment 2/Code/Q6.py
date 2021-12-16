
import numpy as np
import subprocess
import re

# Below function is used to get function value,gradient and hessian at a given point
# Input: point at which function value needs to be obtained
# Output: returns function value, gradient and Q*x0 
def readPoint(SR, currPoint):
    input = str(SR) + ",[" + ','.join(str(elem) for elem in currPoint) + "]"
    someVar = subprocess.run(["Q6_oracle_2.exe", input], stdout=subprocess.PIPE).stdout.decode('utf-8')
    func_val = re.search(r'(.*)', someVar).group(1)
    func_val = float(func_val)
    gr = re.findall(r'\[(.*?)\]',someVar)
    gr = str(gr)
    gr = re.sub('[\'\[\]]', '', gr)
    Arr1 = gr.split(',')
    FloatArr1 = [float(i) for i in Arr1]
    gradient = np.array(FloatArr1[0:10],dtype = np.float64)
    Q_x = np.array(FloatArr1[10:],dtype = np.float64)
    return func_val,gradient,Q_x

## performs exact line search
## Input: x0, flag =1 for 6 f)iii and flag = 2 for 6f)iv . When flag = 2 S_ST can take 3 values corresponding to different beta
## Output: Returns best step size
def exactLineSearch(x0,flag,S_ST):
    
    grad = readPoint(19598,x0)[1] 
    Q_grad = readPoint(19598,grad)[2] ## Returns operation of Q on grad
    Q_grad_mod = readPoint(19598,np.dot(S_ST,grad))[2] ## Returns operation of Q on np.dot(S_ST,grad)
    if flag==1:
        step_size = (np.linalg.norm(grad))**2/(np.dot(grad.T,Q_grad)) ## corresponds to normal gradient descent with exact line search
    else:
        v = np.dot(S_ST,grad)
        step_size = np.dot(grad.T,v)/np.dot(v.T,Q_grad_mod) ## corresponds to scaled gradient descent with exact line search
    return step_size

# Below function performs the actual gradient descent
# Input: SR Number and the initial point 
# Output: Number of iterations required for the Gradient Descent to Converge,point after convergence and the function value 
def gradientDescent(SR, x0,flag,beta):
    iters = 0
    currPoint = x0
    f_val,gradient,Q_x = readPoint(19598,x0)
    S_ST = np.identity(10,dtype=float)
    S_ST[0,0] = beta
    if flag == 1:
        descent_dir  = -gradient
    else:
        descent_dir = -np.dot(S_ST,gradient)
    while np.linalg.norm(gradient) > 0.01:
        step_size = exactLineSearch(currPoint,flag,S_ST)
        currPoint = currPoint + step_size*descent_dir
        ##print("x%d=%s" % (iters, currPoint))
        f_val,gradient,Q_x = readPoint(SR, currPoint)
        if flag ==1:
            descent_dir = -gradient
        else:
            descent_dir = -np.dot(S_ST,gradient)
        iters += 1
    return iters,currPoint,f_val

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    ## part f subpart 3
    iters,currPoint,f_val = gradientDescent(19598,np.array([1,50,50,50,50,50,50,50,50,50]),1,-1)
    print('iters = %d , function value = %s for subpart 3'%(iters,f_val))
    ## part f subpart 4
    iters1,f_val1,currPoint1 = gradientDescent(19598,np.array([1,50,50,50,50,50,50,50,50,50]),2,1/200)
    print('iters = %d , function value = %s,for beta = 1/200'%(iters1,f_val1))
    iters2,f_val2,currPoint2 = gradientDescent(19598,np.array([1,50,50,50,50,50,50,50,50,50]),2,1/700)
    print('iters = %d, function value = %s for beta = 1/700'%(iters2,f_val2))
    iters3,f_val3,currPoint3 = gradientDescent(19598,np.array([1,50,50,50,50,50,50,50,50,50]),2,1/2000)
    print('iters = %d function value = %s for beta = 1/2000'%(iters3,f_val3))
    print('the best beta = 1/700 and corresponding function val is %s and the point to which it converges is'%(f_val2),end = ' ')
    print(currPoint2)