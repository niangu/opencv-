import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier

def loadDataSet():
    ##运行脚本所在目录
    base_dir=os.getcwd()
    ##记得添加header=None，否则会把第一行当作头
    data=pd.read_excel('/home/niangu/Downloads/qq-files/1129701175/file_recv/强度原数据.xlsx')
    ##dataLen行dataWid列 ：返回值是dataLen=100 dataWid=3
    dataLen,dataWid = data.shape
    ##训练数据集
    xList = []
    ##标签数据集
    lables = []
    ##读取数据
    for i in range(dataLen):
        row = data.values[i]
        xList.append(row[0:dataWid-1])
        lables.append(row[-1])
    return xList,lables


def GetResult():
    dataMat,labelMat=loadDataSet()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5,2), random_state=1)
    clf.fit(dataMat, labelMat)
    #print("层数----------------------")
    #print(clf.n_layers_)
    #print("权重----------------------")
    #for cf in clf.coefs_:
    #    print(cf)
    #print("预测值----------------------")
    y_pred=clf.predict(dataMat)
    m = len(y_pred)
    ##分错4个
    t = 0
    f = 0
    for i in range(m):
        if y_pred[i] ==labelMat[i]:
            t += 1
        else :
            f += 1
    print("正确:"+str(t))
    print("错误:"+str(f))

if __name__=='__main__':
    GetResult()