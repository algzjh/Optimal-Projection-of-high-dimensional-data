from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE, Isomap
from . import kda_base
from . import IntegrationCode
from . import calcGONG
import json
import numpy as np
import os
from django.conf import settings

# Create your views here.
"""
def index(request):
    template = loader.get_template('graapp/index.html')
    context = {}
    return HttpResponse(template.render(context, request))
    # return HttpResponse("Hello, world. You're at the index page.")
"""


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def readDataFromFile(filename):
    """
    :param filename: the data file, the format is class_label, dim0, dim1, ...
    :return: data and label
    """
    newfilename = filename  
    data = []
    with open(newfilename, 'r') as f:
        for line in f:
            data.append(list(eval(line)))
    data = np.array(data)
    dim = data.shape[1]
    return np.array(data[:, 1:dim]), np.array(data[:, 0])


def getdistinctnum(a):
    return len(set(a))

def index(request):
    if request.method == 'POST':
        if request.is_ajax() and request.POST['id'] == 'requestPlot':
            # print("1 X_origin: ")
            # X = load_iris().data
            # y = load_iris().target

            # print("2 data: ")
            # print(X)
            print("settings.BASE_DIR: ", settings.BASE_DIR)
            dataName = str(request.POST["dataName"])
            dataName = dataName[6:]
            dataName = 'graapp/' + dataName
            basedir = str(settings.BASE_DIR)
            dataName = os.path.join(basedir, dataName)
            print("dataName2: ", dataName);
            metric = request.POST["originMetric"]
            print("metric: ", metric)
            threshold = float(request.POST["threshold"])
            print("threshold", threshold)
            print("=============")
            print("originData: ")
            X, y = readDataFromFile(dataName)
            y = y.astype(int)
            dataDim = X.shape[1]
            dataSize = X.shape[0]
            dataClass = len(set(y))
            evaluation = request.POST['evaluation']
            # X = np.array(json.loads(request.POST["originData"]))
            # y = np.array(json.loads(request.POST["originLabel"]))
            print(type(X))
            print(type(y))
            print(X)
            print(y)
            print("=============")

            Y_pca = pca(X)
            gong_pca = calcGONG.getGONGValue(Y_pca, y, evaluation)
            Y_mds = mds(X)
            gong_mds = calcGONG.getGONGValue(Y_mds, y, evaluation)
            Y_lda = lda(X, y)
            gong_lda = calcGONG.getGONGValue(Y_lda, y, evaluation)
            Y_isomap = isomap(X)
            gong_isomap = calcGONG.getGONGValue(Y_isomap, y, evaluation)
            Y_tsne = tsne(X)
            gong_tsne = calcGONG.getGONGValue(Y_tsne, y, evaluation)
            Y_pLDA = pLDA(X, y, threshold, metric, evaluation)


            print("tsne: ")
            print(Y_tsne)
            print(Y_tsne.shape)
            print(type(Y_tsne))

            print("Y_pLDA: ")
            print(Y_pLDA)
            print(len(Y_pLDA))
            print(type(Y_pLDA))
            print(len(Y_pLDA))
            print(len(Y_pLDA[0]))
            gong_pLDA = 0
            label_num = getdistinctnum(y)
            

            result = {
                'Y_pca' : formatArray(Y_pca),
                'Y_mds' : formatArray(Y_mds),
                'Y_lda' : formatArray(Y_lda),
                'Y_isomap' : formatArray(Y_isomap),
                'Y_tsne': formatArray(Y_tsne),
                'label' : formatArray(y),
                'Y_pLDA': formatArray(Y_pLDA),
                'dataDim': dataDim,
                'dataSize' : dataSize,
                'dataClass' : dataClass
            }
            view_num = len(Y_pLDA)
            tot_points_num = 0
            result["view_num"] = view_num
            print("view_num: ", view_num)
            for i in range(view_num):
                nx = np.array(Y_pLDA[i][0])
                ny = np.array(Y_pLDA[i][1])
                nl = np.array(Y_pLDA[i][2])
                if i == 0:
                    Y_pLDA1 = np.stack((nx, ny), axis=-1)
                    Y_pLDA1_Label = nl;
                    result["Y_pLDA1"] = formatArray(Y_pLDA1)
                    result["Y_pLDA1_Label"] = formatArray(Y_pLDA1_Label)
                    gong_pLDA += calcGONG.getGONGValue(Y_pLDA1, Y_pLDA1_Label, evaluation) * Y_pLDA1.shape[0]# * getdistinctnum(Y_pLDA1_Label) / label_num
                    tot_points_num += Y_pLDA1.shape[0]
                elif i == 1:
                    Y_pLDA2 = np.stack((nx, ny), axis=-1)
                    Y_pLDA2_Label = nl;
                    result["Y_pLDA2"] = formatArray(Y_pLDA2)
                    result["Y_pLDA2_Label"] = formatArray(Y_pLDA2_Label)
                    gong_pLDA += calcGONG.getGONGValue(Y_pLDA2, Y_pLDA2_Label, evaluation) * Y_pLDA2.shape[0]# * getdistinctnum(Y_pLDA2_Label) / label_num
                    tot_points_num += Y_pLDA2.shape[0]
                elif i == 2:
                    Y_pLDA3 = np.stack((nx, ny), axis=-1)
                    Y_pLDA3_Label = nl;
                    result["Y_pLDA3"] = formatArray(Y_pLDA3)
                    result["Y_pLDA3_Label"] = formatArray(Y_pLDA3_Label)
                    gong_pLDA += calcGONG.getGONGValue(Y_pLDA3, Y_pLDA3_Label, evaluation) * Y_pLDA3.shape[0]# * getdistinctnum(Y_pLDA3_Label) / label_num
                    tot_points_num += Y_pLDA3.shape[0]
                elif i == 3:
                    Y_pLDA4 = np.stack((nx, ny), axis=-1)
                    Y_pLDA4_Label = nl;
                    result["Y_pLDA4"] = formatArray(Y_pLDA4)
                    result["Y_pLDA4_Label"] = formatArray(Y_pLDA4_Label)
                    gong_pLDA += calcGONG.getGONGValue(Y_pLDA4, Y_pLDA4_Label, evaluation) * Y_pLDA4.shape[0]# * getdistinctnum(Y_pLDA4_Label) / label_num
                    tot_points_num += Y_pLDA4.shape[0]
                elif i == 4:
                    Y_pLDA5 = np.stack((nx, ny), axis=-1)
                    Y_pLDA5_Label = nl;
                    result["Y_pLDA5"] = formatArray(Y_pLDA5)
                    result["Y_pLDA5_Label"] = formatArray(Y_pLDA5_Label)
                    gong_pLDA += calcGONG.getGONGValue(Y_pLDA5, Y_pLDA5_Label, evaluation) * Y_pLDA5.shape[0]# * getdistinctnum(Y_pLDA5_Label) / label_num
                    tot_points_num += Y_pLDA5.shape[0]
            result["gong_pca"] = gong_pca
            result["gong_mds"] = gong_mds
            result["gong_lda"] = gong_lda
            result["gong_isomap"] = gong_isomap
            result["gong_tsne"] = gong_tsne
            result["gong_pLDA"] = gong_pLDA / tot_points_num
            return HttpResponse(json.dumps(result, cls=NpEncoder))
        return HttpResponse(json.dumps({'message' : 'error'}))
    return render(request, 'graapp/index.html')



def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close

def string2float(X):
    for i in range(len(X)):
        me = X[i]
        for j in range(len(me)):
            X[i][j] = float(X[i][j])
    return X


# 求向量
def eucldist_forloop(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    dist = 0
    for (x, y) in zip(coords1, coords2):
        dist += (float(x) - float(y)) ** 2
    return dist ** 0.5


def pca(X):
    pca = PCA(n_components = 2)
    pca.fit(X)
    X_new = pca.transform(X)
    return X_new

def mds(X):
    mds = MDS(n_components=2)
    X_new = mds.fit_transform(X)
    return X_new

def lda(X, y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    print("#######################################")
    print(X_new)
    if X_new.shape[1] == 1:
        X_new = np.c_[X_new[:,0], X_new[:, 0]]
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(X_new)
    return X_new

def tsne(X):
    tsne = TSNE(n_components=2)
    X_new = tsne.fit_transform(X)
    return X_new

def kda(X, y):
    kda = kda_base.KernelDiscriminantAnalysis()
    a = kda.fit(X, y)
    b = kda.transform(X)
    return b
    # return b[:, 0:2]

def pLDA(X, y, threshold, measure_option, evaluation):
    result_list = IntegrationCode.getpLDA_pass(X, y, threshold, measure_option, evaluation)
    return result_list

def isomap(X):
    embedding = Isomap(n_components=2)
    X_transformed = embedding.fit_transform(X)
    return X_transformed

def formatArray(array):
    ret = list(np.array(array).tolist())
    return ret



