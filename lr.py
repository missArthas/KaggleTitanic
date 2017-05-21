# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

data_train = pd.read_csv('data/train.csv')
print('data_train:',data_train.info())

# fig = plt.figure()
# fig.set(alpha=0.2)
#
# plt.subplot2grid((2,3),(0,0))
# data_train.Survived.value_counts().plot(kind='bar')
# plt.title(u'获救情况（1为获救）')
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.title(u'乘客等级分布')
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived,data_train.Age)
# plt.ylabel(u'年龄')
# plt.grid(b=True, which='major', axis='y')
# plt.title(u'按年龄看获救分布(1为获救)')
#
# plt.subplot2grid((2,3),(1,0),colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.title(u'各等级的乘客年龄分布')
# plt.legend((u'头等舱',u'2等舱',u'3等舱'),loc='best')
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u'各登船口岸上船人数')
# plt.ylabel(u'人数')
# plt.show()
#
#
# fig = plt.figure()
# fig.set(alpha=0.2)
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df = pd.DataFrame({u'男性':Survived_m,u'女性':Survived_f})
# precessed_train.plot(kind='bar',stacked=True)
# plt.title(u'按性别来看获救情况')
# plt.xlabel(u'性别')
# plt.ylabel(u'人数')
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.65)
# plt.title(u'根据舱等级和性别的获救情况')
#
# ax1 = fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',label='female highclass', color='#FA2479')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'女性/高级舱'],loc='best')
#
# ax2 = fig.add_subplot(142, sharey = ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',label='female lowclass', color='pink')
# ax2.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'女性/低级舱'],loc='best')
#
# ax1 = fig.add_subplot(143, sharey = ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',label='female highclass', color='lightblue')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'男性/高级舱'],loc='best')
#
# ax2 = fig.add_subplot(144, sharey = ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',label='female lowclass', color='steelblue')
# ax2.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'男性/低级舱'],loc='best')
#
#
# fig = plt.figure()
# fig.set(alpha=0.2)
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
# precessed_train.plot(kind='bar',stacked=True)
# plt.title(u'各登录港口的获救情况')
# plt.xlabel(u'登录港口')
# plt.ylabel(u'人数')
# plt.show()
#
# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df
#
# g = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df
#
# data_train.Cabin.value_counts()
#
# fig = plt.figure()
# fig.set(alpha=0.2)
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df = pd.DataFrame({u'有':Survived_cabin,u'无':Survived_nocabin})
# precessed_train.plot(kind='bar',stacked=True)
# plt.title(u'按Cabin有无看获救情况')
# plt.xlabel(u'Cabin有无')
# plt.ylabel(u'人数')
# plt.show()

def set_missing_ages(df):
    
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    
    known_age = age_df[age_precessed_train.Age.notnull()].as_matrix()
    unknown_age = age_df[age_precessed_train.Age.isnull()].as_matrix()

    y = known_age[:,0]
    
    X = known_age[:,1:]
    
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    
    predictedAges = rfr.predict(unknown_age[:,1::])
    
    precessed_train.loc[(precessed_train.Age.isnull()),'Age'] = predictedAges
           
    return df,rfr

def set_Cabin_type(df):
    
    precessed_train.loc[ (precessed_train.Cabin.notnull()),'Cabin' ] = 'Yes'
    precessed_train.loc[ (precessed_train.Cabin.isnull()),'Cabin' ] = 'No'
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Cabin = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix = 'Pclass')

df = pd.concat([data_train,dummies_Cabin,dummies_Cabin,dummies_Sex,dummies_Pclass],axis=1)
precessed_train.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)


train_df = precessed_train.filter(regex='Survived|Age_.*|Sibsp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass_*')
train_np = train_precessed_train.as_matrix()

y = train_np[:,0]

X = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='11',tol=1e-6)
clf.fit(X,y)

data_test = pd.read_csv('test.csv')
data_test.loc[(data_test.Fare.isnull()),'Fare']=0
              
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix() 

X = null_age[:,1:]
    
predictedAges = rfr.predict(X)

data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Cabin = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'],prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix = 'Pclass')

data_test = pd.concat([data_test,dummies_Cabin,dummies_Cabin,dummies_Sex,dummies_Pclass],axis=1)
data_test.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

data_test['Age_scaled'] = scaler.fit_transform(data_test['Age'],age_scale_param)
data_test['Fare_scaled'] = scaler.fit_transform(data_test['Fare'],fare_scale_param)


test = data_test.filter(regex='Age_.*|Sibsp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass_*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix,'Survived':predictions.astype(np.int32)})
result.to_csv('logistic_regression_predictions.csv',index=False)
