# https://developer.aliyun.com/article/73661
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


###################  sigmoid  #####################

def sigmoid(inputs):
    """
    Calculate the sigmoid for the give inputs (array)
    :param inputs:
    :return:
    """
    sigmoid_scores = [1 / float(1 + np.exp(-x)) for x in inputs]
    return sigmoid_scores


sigmoid_inputs = [2, 3, 5, 6]
print("Sigmoid Function Output :: {}".format(sigmoid(sigmoid_inputs)))


def line_graph(x, y, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x:
    :param y:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


graph_x = range(0, 21)
graph_y = sigmoid(graph_x)

print("Graph X readings: {}".format(graph_x))
print("Graph Y readings: {}".format(graph_y))

line_graph(graph_x, graph_y, "Inputs", "Sigmoid Scores")


###################  softmax  #####################

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))


softmax_inputs = [2, 3, 5, 6]
print("Softmax Function Output :: {}".format(softmax(softmax_inputs)))


graph_x = range(0, 21)
graph_y = softmax(graph_x)

print("Graph X readings: {}".format(graph_x))
print("Graph Y readings: {}".format(graph_y))

line_graph(graph_x, graph_y, "Inputs", "Softmax Scores")



"""
Sigmoid函数与Softmax函数之间的差异
以下是Sigmoid和Softmax函数之间的差异表格：


确定逻辑回归模型的两个函数。

Softmax：用于多分类任务。
Sigmoid：用于二进制分类任务。

"""