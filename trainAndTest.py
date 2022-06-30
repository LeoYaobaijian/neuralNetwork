'''
Descripttion: 
version: 
Author: Yao BaiJian
Date: 2022-06-29 18:13:13
LastEditors: Andy
LastEditTime: 2022-06-30 10:38:10
'''
from ast import increment_lineno
import numpy as np
import matplotlib.pyplot
from NetworkClass import neuralNetworkClass


def trainTheNetwork(input_nodes = 100,
                    hidden_nodes = 100,
                    output_nodes = 10,
                    learning_rate = 0.3):

    trainedNet = neuralNetworkClass(input_nodes, hidden_nodes, output_nodes, learning_rate)
    data_file = open("MNIST CSV/mnist_train.csv", 'r')
    training_data_list = data_file.readlines()
    data_file.close()
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        trainedNet.train(inputs, targets)
    return trainedNet

trainedNet = trainTheNetwork(784, 100, 10, 0.3)

test_data_file = open("MNIST CSV/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = trainedNet.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() /scorecard_array.size)


