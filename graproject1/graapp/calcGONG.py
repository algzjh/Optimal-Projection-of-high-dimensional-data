import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import math


def getDistance2Point(P1, P2):
    dx = abs(P1[0] - P2[0])
    dy = abs(P1[1] - P2[1])
    return math.sqrt(dx * dx + dy * dy)


def calcOnePointGONG(Points, Label, i, VDict):
    omega = 0
    same_indicator = 0
    tot_num = Points.shape[0]
    for j in range(tot_num):
        if j == i:
            continue
        intermediary_point = 0.35 * Points[j] + 0.65 * Points[i]
        int2j_dist = getDistance2Point(intermediary_point, Points[j])
        if VDict[j][0][1] != i:
            if int2j_dist < VDict[j][0][0]:
                omega += 1
                if Label[i] == Label[j]:
                    same_indicator += 1
        elif int2j_dist < VDict[j][1][0]:
            omega += 1
            if Label[i] == Label[j]:
                same_indicator += 1
    if omega == 0:
        return 0
    return same_indicator/omega


def preCalcVDict(Points):
    VDict = []
    tot_num = Points.shape[0]
    for i in range(tot_num):
        now_dict = {0: (math.inf, 0), 1: (math.inf, 0)}  # 最近 次近 （距离， id）
        for j in range(tot_num):
            if j == i:
                continue
            ijdist = getDistance2Point(Points[i], Points[j])
            if ijdist < now_dict[0][0]:
                now_dict[1] = now_dict[0]
                now_dict[0] = (ijdist, j)
            elif ijdist < now_dict[1][0]:
                now_dict[1] = (ijdist, j)
        VDict.append(now_dict)
    return VDict


def getGONGSepList(Points, Label):

    gongSepList = []
    tot_num = Points.shape[0]
    VDict = preCalcVDict(Points)
    for i in range(tot_num):
        gongSepList.append(calcOnePointGONG(Points, Label, i, VDict))
    return gongSepList


def getGONGValue(Points, Label, evaluation):
    if evaluation == "Silhouette Coefficient":
        return metrics.silhouette_score(Points, Label)
    gongSepList = getGONGSepList(Points, Label)
    labelcount = {}
    labelSep = {}
    tot_num = Points.shape[0]
    for i in range(tot_num):
        if Label[i] not in labelcount.keys():
            labelcount[Label[i]] = 1
            labelSep[Label[i]] = gongSepList[i]
        else:
            labelcount[Label[i]] += 1
            labelSep[Label[i]] += gongSepList[i]
    answer = 0
    for key, value in labelSep.items():
        answer += value / labelcount[key]
    answer /= len(labelcount)
    return answer


def readDataFromFile(filename):
    """
    :param filename: the data file, the format is class_label, dim0, dim1, ...
    :return: data and label
    """
    newfilename = './Data/' + filename  # because my data is in the Data folder
    data = []
    with open(newfilename, 'r') as f:
        for line in f:
            data.append(list(eval(line)))
    data = np.array(data)
    dim = data.shape[1]
    return data[:, 1:dim], data[:, 0]

"""

fileName = "minist.csv"
# data, target = readDataFromFile(fileName)
data, target = datasets.load_wine(return_X_y=True)
target = target.astype(int)

pca = PCA(n_components=2)
new_data_PCA = pca.fit_transform(data)

colors = ["grey", "brown", "orange", "olive", "green", "cyan",
          "blue", "purple", "pink", "red"]
plt.scatter(new_data_PCA[:, 0], new_data_PCA[:, 1], c=target,
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

clf = LDA(n_components=2)
new_data_LDA = clf.fit_transform(data, target)
plt.scatter(new_data_LDA[:, 0], new_data_LDA[:, 1], c=target,
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

PCA_GONG = getGONGValue(new_data_PCA, target)
print("PCA_GONG: ", PCA_GONG)
LDA_GONG = getGONGValue(new_data_LDA, target)
print("LDA_GONG: ", LDA_GONG)
"""
