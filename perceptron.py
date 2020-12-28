# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:09:29 2020

@author: yashv
"""

import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=5, learning_rate=100000):
        self.threshold= threshold
        self.learning_rate= learning_rate
        self.weights= np.zeros(no_of_inputs+1)
        
    def predict(self,inputs):
        summation= np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation>0:
            activation=1
        else:
            activation=0
        return activation
        
    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            print("Epoch_no:",_)
            print("Weights:",self.weights)
            for inputs, label in zip(training_inputs, labels):
                prediction= self.predict(inputs)
                self.weights[1:] += self.learning_rate * inputs * (label-prediction)
                self.weights[0] += self.learning_rate * (label-prediction)