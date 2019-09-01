# coding = utf-8

import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
myfont = fm.FontProperties(fname = 'C:\Windows\Fonts\simsun.ttc')

def create_dateset():
    datasets = array([[8, 4, 2], [7, 1, 1], [1, 4, 4], [3, 0, 5]])
    labels = ['非常热', '非常热', '一般热', '一般热']
    return datasets, labels

def analyze_data_plot(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    # plt.scatter(x, y)
    plt.title('有课冷热感知点散点图', fontsize = 25, fontname = '宋体', fontproperties = myfont)
    plt.xlabel('天热吃冰淇淋数目', fontsize = 15, fontname = '宋体', fontproperties = myfont)
    plt.ylabel('天热喝水数目', fontsize = 15, fontname = '宋体', fontproperties = myfont)
    plt.show()

def knn_Classifier(newV, datasets, labels, k):
    SqrtDist = EuclideanDistance3(newV, datasets)
    sortDistIndexs = SqrtDist.argsort(axis = 0)

# def ComputeEuclideanDistance(x1, y1, x2, y2):
#     d = math.sqrt(math.pow((x1 - x2), 2) + (math.pow((y1 - y2), 2))
#     return d

def EuclideanDistance(instance1, instance2, length):
    d = 0
    for x in range(length):
        d += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(d)

def EuclideanDistance3(newV, datasets):
    rowsize, colsize = datasets.shape
    diffMat = tile(newV, (rowsize, 1)) - datasets
    sqDiffMat = diffMat ** 2
    SqrtDist = sqDiffMat.sum(axis = 1) ** 0.5
    return SqrtDist

if __name__ == '__main__':
    datasets, labels =  create_dateset()
    print('数据集\n', datasets, '类标签\n', labels)
    # analyze_data_plot(datasets[:, 0], datasets[:, 1])

    # d = ComputeEuclideanDistance(2, 4, 8, 2)
    # print(d)

    d2 = EuclideanDistance([2, 4, 4], [7, 1, 1], 3)
    print(d2)

    d3 = EuclideanDistance3([2, 4, 4], datasets)
    print(d3)

    knn_Classifier([2, 4, 4], datasets, labels, 2)

    # newV = [2, 4, 0]
    # knn_Classifier(newV, datasets, labels, 2)