
# coding: utf-8

# In[ ]:


# Make a prediction with weights
from math import exp
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
train=[]   ###dataset for training
for i in range(0,10):
    train.append(x[i])
pred=arr[:,:]      ##prediction data set
y = np.array(t)
print("This is my class labels where A=-1 and B=1")
print(y)   ###class labels
dataset = train
lr = 0.7
itera = 100

# Estimate logistic regression coefficients using stochastic gradient descent
def logistic(train, lr, itera):
    w = [0.0 for i in range(len(train[0]))]
    for iteraa in range(itera):
        sum_error = 0
        for row in train:
            yt = predict(row, w)
            error = row[-1] - yt
            sum_error =sum_error+ error**2
            w[0] = w[0] + lr * error * yt * (1.0 - yt)
            for i in range(len(row)-1):
                w[i + 1] = w[i + 1] + lr * error * yt * (1.0 - yt) * row[i]
       
    return w
def predict(row, weights):
    yt = weights[1]
    for i in range(len(row)-1):
        yt += weights[i + 1] * row[i]
    result=1.0 / (1.0 + exp(-yt))
    return result
 
# Calculate weights by training on data set
w=logistic(dataset, lr, itera)

##prediction for all the labels
for i in range(0,5):
    print(pred[i])
    predict(pred[i],w)
    if predict(pred[i],w)<0.2:
        print("A")
    else:
        print("B")

