# coding=utf-8
# created by czc on 2017.7.26
import os.path

from LDA_module import LDA_module
from LSA_module import LSA_module

rootdir_train = "G:\PyCharmWorkSpace/train_set"                   # 指明被遍历的文件夹
rootdir_predict = "G:\PyCharmWorkSpace/train_set"                   # 指明被遍历的文件夹
train_set = []
predict_set = []

# 获取训练集
for parent, dirnames, filenames in os.walk(rootdir_train):        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    for filename in filenames:
        # print("filename is:", filename)
        fileHandle = open(rootdir_train+'/'+filename, 'r')
        doc = fileHandle.readlines()
        for line in doc:
            train_set.append(line)
        fileHandle.close()
# print("训练集：\n", train_set)

# 获取待测测试集
for parent_no_use, dirnames_no_use, filenames in os.walk(rootdir_predict):        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    for filename in filenames:
        # print("filename is:", filename)
        fileHandle = open(rootdir_predict+'/'+filename, 'r')
        doc = fileHandle.read()
        predict_set.append(doc)
        fileHandle.close()
# print("测试集：\n", predict_set)

print("训练LDA模型并作主题分析：")
instance_ldamodule = LDA_module(train_set, num_topics=100, num_words=6, num_traversals=20)
instance_ldamodule.trainmodel()
instance_ldamodule.predict(predict_set)

print("训练LSA模型并作主题分析：")
instance_lsamodule = LSA_module(train_set, num_topics=100, num_words=6, num_traversals=20)
instance_lsamodule.trainmodel()
instance_lsamodule.predict(predict_set)
