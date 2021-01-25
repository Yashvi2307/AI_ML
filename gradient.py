# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:07:32 2021

@author: yashv
"""

import numpy as np

X= [0.7, 1.5]
Y= [3.9,0.2]

def f(w,b,x): #sigmoid logistic function
    return 1.0/(1.0 + np.exp(-(w*x +b)))

def error(w,b): #loss function
    err=0.0
    for x,y in zip(X,Y):
        fx= f(w,b,x)
        err += 0.5 * (fx - y) **2
    return err

def grad_b(w,b,x,y):
    fx= f(w,b,x)
    return (fx - y)* fx * (1-fx) 

def grad_w(w,b,x,y):
    fx= f(w,b,x)
    return (fx - y)* fx * (1-fx) * x

def do_gradient_descent():
    w, b, eta, max_epochs = 10, 10, 6.0, 1000
    for i in range(max_epochs):
        dw, db = 0,0 
        for x,y in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - eta * dw
        b = b - eta * dw
        print(w,b)
    print("e:",error(w,b))
    
do_gradient_descent()



