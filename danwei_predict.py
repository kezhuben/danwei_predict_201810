# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:40:28 2018

@author: 竹本
"""
from numpy import *
import operator
from os import listdir
from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data file
data0 = pd.read_csv("D://Sunmnet//Employment_trend_predict//danweixinzhi//dangwei_stu_info_data5.csv",encoding = 'utf8')

data = data0.drop(['入学年份','major_code'],axis = 1)
data["minzu"] = data["minzu"].map(lambda x: 1 if (x == "汉族") else 0)
#data['hkxz'] = data['hkxz'].fillna('农村')
data["hkxz"] = data["hkxz"].map(lambda x: 1 if (x == "城镇") else 0)

data["SHENGYUANDI"] = data["SHENGYUANDI"].map(lambda x: 
        1 if x == "西部地区" else (
        2 if x == "陕西省" else (
        3 if x == "西部地区" else 
        4 if x == "东北地区" else 
        5 if x == "中部地区" else 6)))
    
data["workplace"] = data["workplace"].map(lambda x: 
        1 if x == "西部地区" else (
        2 if x == "陕西省" else (
        3 if x == "西部地区" else 
        4 if x == "东北地区" else 
        5 if x == "中部地区" else 6)))
    
data["ZHENGZHIMIANMAO"] = data["ZHENGZHIMIANMAO"].map(lambda x: 
        1 if x == "共产党" else (
        2 if x == "共青团" else 3))
 
data['college_code'] = data['college_code'].replace(
        "医学院","医学").replace(
        "医学部","医学").replace(
        "生命科学与技术学院","医学")

data['college_code'] = data['college_code'].replace(
        "公共政策与管理学院","文史").replace(
        "人文社会科学学院","文史").replace(
        "法学院","文史").replace(
        "外国语学院","文史").replace(
        "管理学院","文史")

data['college_code'] = data['college_code'].replace(
        "机械工程学院","工学").replace(
        "航天航空学院","工学").replace(
        "食品装备工程与科学学院","工学").replace(
        "软件学院","工学").replace(
        "能源与动力工程学院","工学").replace(
        "电子与信息工程学院","工学").replace(
        "化学工程与技术学院","工学").replace(
        "材料科学与工程学院","工学").replace(
        "电气工程学院","工学")

data['college_code'] = data['college_code'].replace(
        "数学与统计学院","理学").replace(
        "理学院","理学")

data['college_code'] = data['college_code'].replace(
        "经济与金融学院","经济学").replace(
        "金禾经济研究中心","经济学")

data['college_code'] = data['college_code'].replace(
        "人居环境与建筑工程学院","艺术")

data["college_code"] = data["college_code"].map(lambda x: 
        1 if x == "医学" else (
        2 if x == "文史" else (
        3 if x == "工学" else 
        4 if x == "理学" else 
        5 if x == "经济学" else 6)))
    
data['单位性质'] = data['单位性质'].replace(
        "国有企业",1).replace(
        "机关",1).replace(
        "部队",1).replace(
        "其他企业",2).replace(
        "医疗卫生单位",2).replace(
        "中初教育单位",2).replace(
        "高等教育单位",2).replace(
        "三资企业",2).replace(
        "其他事业单位",2).replace(
        "科研设计单位",2).replace(
        "其他",3).replace(
        "自主创业",3).replace(
        "自由职业",3)
 
data= data.fillna(0.0)
#data = data.ix[data.sy_salary > 700]
"""
data["单位性质"] = data["单位性质"].map(lambda x: 
        1 if x == "国企事业单位" else (
        2 if x == "一般企业" else
        3))
"""    

"""Data normalization"""
#data_dummies_norm = data_dummies.ix[:,'sy_salary':'XFCJ'].apply(lambda x:(x - np.min(x)) / (np.max(x)-np.min(x)))
data_dummies_norm = data.ix[:,'sy_salary':'col_7'].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)))


data_dummies_all_data = data[['xh','单位性质','college_code','minzu', 'SHENGYUANDI', 
                              'ZHENGZHIMIANMAO', 'workplace', 'hkxz', 'have_fx']].join(data_dummies_norm)
#data_dummies_all_data = data_dummies_all_data.drop(['col_7','col_5','col_6', 'have_fx', 'col_3', '生源地代码', '政治面貌','民族','hkxz'],axis = 1)
#data_dummies_all_data = data_dummies_all_data.drop(['XFCJ','生源地代码', '政治面貌','民族','hkxz'],axis = 1)

data_dummies_all_data = data_dummies_all_data.drop(['SHENGYUANDI', 'hkxz','minzu','ZHENGZHIMIANMAO'],axis =1)
from sklearn.model_selection import train_test_split
trainData, testData = train_test_split(data_dummies_all_data, test_size=0.2,random_state=888)

train_cols = trainData.columns[2:]
test_cols = testData.columns[2:]

trainData_X = trainData[train_cols]
trainData_Y = trainData[["单位性质"]]

testData_X = testData[test_cols]
testData_Y = testData[["单位性质"]]

# 特征权重
train_cols = trainData_X.columns
importances = model.feature_importances_
mportances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(trainData_X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, train_cols[indices[f]], importances[indices[f]]))

"""knn"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(trainData_X,trainData_Y)
knn.score(testData_X,testData_Y) #0.71
#test_predict = knn.predict(testData_X)

"""tree"""
from sklearn.tree import DecisionTreeClassifier
#decision=DecisionTreeClassifier(max_depth=5)
model_tree = DecisionTreeClassifier(criterion="entropy",  # 切割标准,可选信息熵(entropy),与基尼不纯度(gini)
                               max_depth=5,  # 决策树的最大深度，默认为None
                               min_samples_split=5,  # 每次分裂节点是最小的分裂个数，即最小被分裂为几个，默认为2
                               min_samples_leaf=5,  # 若某一次分裂时一个叶子节点上的样本数小于这个值，则会被剪枝，默认为1
                               max_features="auto",
                               random_state=888,
                               # 每一次生成树时使用的特征数量,默认"auto"(无限制),可指定"sqrt","log2",或者是int(指定数字),float(指定占比)
                               max_leaf_nodes=None,  # 最大的叶子节点的个数，默认为None,如果不为None，max_depth参数将被忽略
                               class_weight={1:5,2:3,3:1})  # 样本权重,应对样本不平衡问题的另一种方法,指定类似{0: 1, 2: 1},则0类的惩罚因子比1类的高一倍,使得错分0类的代价更大

#model_tree = DecisionTreeClassifier()
model_tree.fit(trainData_X,trainData_Y)
model_tree.score(testData_X,testData_Y)
test_predict = model_tree.predict(testData_X)

# 组合
F = pd.DataFrame({"pred": test_predict, "real": testData_Y['单位性质']})
F.head()

from sklearn.metrics import confusion_matrix

confusion_matrix(F.real, F.pred)

# 类别的准确率、召回率和F-值
def my_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    print("classification_report(left: labels):")
    print(classification_report(y_true, y_pred))

my_classification_report(F.real, F.pred)

# 可视化树图
#data_ = pd.read_csv("mushrooms.csv")
data_feature_name = trainData.columns[2:]
data_target_name = np.unique(trainData["单位性质"].astype(str))
import graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin'
dot_tree = tree.export_graphviz(model_tree,out_file=None,feature_names=data_feature_name,class_names=data_target_name,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_tree)
img = Image(graph.create_png())
#graph.write_png("D://Sunmnet//Employment_trend_predict//danweixinzhi//out2.png")


"""SVM"""
from sklearn.svm import SVC
svc = SVC(kernel='rbf', decision_function_shape='ovr',class_weight='balanced')
svc.fit(trainData_X,trainData_Y)
svc.score(testData_X,testData_Y) # 0.75
#test_predict = svc.predict(testData_X)


"""forest"""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=2000,  # 指定树的数量,建议足够的大,从本例看5棵树与500棵树效果差异非常大
                               criterion="entropy",  # 切割标准,可选信息熵(entropy),与基尼不纯度(gini)
                               max_depth=5,  # 决策树的最大深度，默认为None
                               min_samples_split=5,  # 每次分裂节点是最小的分裂个数，即最小被分裂为几个，默认为2
                               min_samples_leaf=5,  # 若某一次分裂时一个叶子节点上的样本数小于这个值，则会被剪枝，默认为1
                               max_features="auto",
                               # 每一次生成树时使用的特征数量,默认"auto"(无限制),可指定"sqrt","log2",或者是int(指定数字),float(指定占比)
                               max_leaf_nodes=None,  # 最大的叶子节点的个数，默认为None,如果不为None，max_depth参数将被忽略
                               bootstrap=True,  # 是否使用bootstrap进行抽样以应对样本不平衡问题
                               n_jobs=3,  # 指定在多少个cup上并行运行
                               class_weight={1:4,2:2,3:1})  # 样本权重,应对样本不平衡问题的另一种方法,指定类似{0: 1, 2: 1},则0类的惩罚因子比1类的高一倍,使得错分0类的代价更大


model.fit(trainData_X,trainData_Y)
model.score(testData_X,testData_Y)
test_predict = model.predict(testData_X)

# 组合
F = pd.DataFrame({"pred": test_predict, "real": testData_Y['单位性质']})
F.head()

from sklearn.metrics import confusion_matrix

confusion_matrix(F.real, F.pred)

# 类别的准确率、召回率和F-值
def my_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    print("classification_report(left: labels):")
    print(classification_report(y_true, y_pred))

my_classification_report(F.real, F.pred)



"""
# 预测
#pred_all = pd.DataFrame(model.predict(data[train_cols]),index=data[train_cols].index)

pro_all = pd.DataFrame(model.predict_proba(testData_X),index=testData_X.index)
pro_all.columns = ["pro_1", "pro_2", "pro_3"]

# 组合
k = 0.3
all_pred = pd.DataFrame({"pred": test_predict, "real": testData_Y['单位性质'], "pro_1": pro_all["pro_1"], "pro_2": pro_all["pro_2"],"pro_3": pro_all["pro_3"]})
all_pred.head()

all_pred["pred_chance"] = all_pred["pro_1"].map(lambda x: 1 if x > k else 2)
all_pred.index = all_pred['index']

my_classification_report(all_pred["real"], all_pred["pred_chance"])
confusion_matrix(all_pred["real"], all_pred["pred_chance"])
"""


"""    
#过度抽样处理库SMOTE
x=data.iloc[:,2:]
y=data.iloc[:,1]

import pandas as pd
from imblearn.over_sampling import SMOTE    
groupby_data_orginal=data.groupby('单位性质').count()   

model_smote=SMOTE()    #建立smote模型对象
x_smote_resampled,y_smote_resampled=model_smote.fit_sample(x,y)
x_smote_resampled=pd.DataFrame(x_smote_resampled,columns=['民族', '生源地代码', '政治面貌', '工作地代码', 'hkxz', 'have_fx', '试用月薪',
       'college_code', 'major_code', 'XFCJ', 'score_min', 'score_max',
       'score_mean', 'major_score', 'language_score', 'by_score', 'col_0',
       'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7'])
y_smote_resampled=pd.DataFrame(y_smote_resampled,columns=['单位性质'])
smote_resampled=pd.concat([x_smote_resampled,y_smote_resampled],axis=1)
groupby_data_smote=smote_resampled.groupby('单位性质').count()

#欠抽样
from imblearn.under_sampling import RandomUnderSampler 
model_RandomUnderSampler=RandomUnderSampler()                #建立RandomUnderSample模型对象
x_RandomUnderSample_resampled,y_RandomUnderSample_resampled=model_RandomUnderSampler.fit_sample(x,y)         #输入数据并进行欠抽样处理
x_RandomUnderSample_resampled=pd.DataFrame(x_RandomUnderSample_resampled,columns=['民族', '生源地代码', '政治面貌', '工作地代码', 'hkxz', 'have_fx', '试用月薪',
       'college_code', 'major_code', 'XFCJ', 'score_min', 'score_max',
       'score_mean', 'major_score', 'language_score', 'by_score', 'col_0',
       'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7'])
y_RandomUnderSample_resampled=pd.DataFrame(y_RandomUnderSample_resampled,columns=['单位性质'])
RandomUnderSampler_resampled=pd.concat([x_RandomUnderSample_resampled,y_RandomUnderSample_resampled],axis=1)
groupby_data_RandomUnderSampler=RandomUnderSampler_resampled.groupby('单位性质').count()

"""

