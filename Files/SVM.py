
# coding: utf-8

# In[1]:


import numpy as np
import numpy 
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
dt=pd.read_csv('data.csv',sep=',',encoding='cp1252')
outing=pd.read_csv('out.csv',sep=',',header=0,encoding='cp1252')
arr=outing.values
array=dt.values
X=dt.drop('clas',axis=1)
x=array[:,1:]
t=array[:,0]
train=[]
X = np.array([
    x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]
])
for i in range(0,10):
    train.append(x[i])
pred=arr[:,:]
y = np.array(t)
print("This is my class labels where A=-1 and B=1")
print(y)   ###class labels
   ###input dataset
def svm(X, Y):
    ##weight vector based on length of my date set which is 25
    w = np.zeros(len(X[0]))
    ##learning rate
    lr = 1     ## i have defined learning rate to be 1 based on results
    ##number of iterations
    itera = 100000
    ##i am just storing errors which i plotted versus iterations 
    errors = []

    ##I have applied gradient descent part and checking every iteration 
    for itera in range(1,itera):
        error = 0
        for i, x in enumerate(X):
                                                 ###Not correctly classified
            if (Y[i]*np.dot(X[i], w)) < 1:
                ##update weights
                w = w + lr * ( (X[i] * Y[i]) + (-2  *(1/itera)* w) )
                error = 1
            else:
               ##update weights
                w = w + lr * (-2  *(1/itera)* w)
        errors.append(error)    
    return w,errors
def prediction(X):
    for i, x in enumerate(X):
                #misclassification
                #print((y[i]*np.dot(X[i], w)))
                print(X[i])
                if (y[i]*np.dot(X[i], w)) <0:

                    print("B")
                else:
                    print("A")
w,errors = svm(X,y)
X=pred
prediction(X)

##i have plotted errors for wrongly classified
plt.plot(errors, "|",color="red")
plt.xlabel('iterations')
plt.ylabel('wrong predictions')
plt.show()

