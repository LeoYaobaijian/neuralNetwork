'''
Descripttion: neuralNetworkClass / three layers
version: 
Author: Yao BaiJian
Date: 2022-06-29 09:23:33
LastEditors: Andy
LastEditTime: 2022-06-29 15:47:10
'''

import numpy as np
import scipy.special


class neuralNetworkClass:
    
    def __init__(self, 
                 inNodes = 3, 
                 hideNodes = 3,
                 outNodes = 3, 
                 learnRate = 0.5):
        
        self.inNodes = inNodes
        self.hideNodes = hideNodes
        self.outNodes = outNodes
        self.lr = learnRate
        
        self.inHideW = (np.random.normal(0.0, pow(self.hideNodes, -0.5), (self.hideNodes, self.inNodes)))
        self.hideOutW = (np.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hideNodes)))
        
        self.activeFunc = lambda x: scipy.special.expit(x)
        
    # train the network
    def train(self, inList, tarList):
        inputs = np.array(inList, ndmin=2).T
        targets = np.array(tarList, ndmin=2).T
        
        hidden_inputs = np.dot(self.inHideW, inputs)
        hidden_outputs = self.activeFunc(hidden_inputs)
        final_inputs = np.dot(self.hideOutW, hidden_outputs)
        final_outputs = self.activeFunc(final_inputs)

        finalError = targets - final_outputs
        hideError = np.dot(self.hideOutW.T, finalError)
    
        self.hideOutW += self.lr * np.dot( (finalError * final_outputs *(1.0 - final_outputs)) , np.transpose(hidden_outputs))
        self.inHideW += self.lr * np.dot(( hideError * hidden_outputs *(1.0 - hidden_outputs)), np.transpose(inputs))
    
    #query the network
    def query(self, inList):
        inputs = np.array(inList, ndmin=2).T
        hideIn = np.dot(self.inHideW, inputs)
        hideOut = self.activeFunc(hideIn)
        finalIn = np.dot(self.hideOutW, hideOut)
        finalOut = self.activeFunc(finalIn)
        return finalOut
        