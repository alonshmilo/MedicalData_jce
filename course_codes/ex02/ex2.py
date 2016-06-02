#Alon Shmilovich, id 034616359, alonsh, Jerusalem College of Engineering JCE

import matplotlib.pyplot as plt
import numpy as np
import copy

#Initiailize:
x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
z = np.array([[1],[1],[1],[0]])

weights = [0,0,0] #Here the right weights will be held
prev_weights=[0,0,0] #For comparisons

i=0 #For iterations on matrix x
num=0 #Bounds
n=0 #Network for output
counter=0 #Counter for comparisons
threshold = 0.5 #This threshold will help us decide n
learning_rate = 0.1 #Learning correction
c=np.array([0,0,0]) #Results for x * weights

while num<100: #Set an upper bound
    if i>=4: #4 iterations for matrix x
        i=0

    c = x[i, 0:3] * weights

    sum=np.sum(c)

    if (sum > threshold):
        n=1
    else:
        n=0

    error = int(z[i] - n)

    correction = learning_rate * error

    prev_weights = copy.copy(weights)

    weights += x[i,0:3] * correction

    if (prev_weights == weights).all():
        counter+=1
        if counter==3:
            break
    else:
        counter = 0

    print "Weights # ",num, weights

    i+=1
    num+=1

    y1 = (-weights[1]*3 - weights[0])/(weights[2] - threshold)
    print "First point: (3,",y1,")"

    y2 = (-weights[1]*(-2) - weights[0])/(weights[2] - threshold)
    print "First point: (-2,",y2,")"

    X=[3,-2]
    Y=[y1,y2]

    plt.title('NAND Perceptron')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.plot([0, 0, 1, 1], [0, 1, 0, 1], 'ro', ms=10)
    plt.axis([-1, 2, -1, 2])
    plt.plot([1], [1], 'bo', ms=10)
    plt.plot(X, Y)
    plt.show()
