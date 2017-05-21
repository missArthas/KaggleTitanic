import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold
import numpy as np
import sklearn.preprocessing as preprocessing


titanic_train=pandas.read_csv("data/train.csv")
titanic_test=pandas.read_csv("data/test.csv")

#train precess
dummies_Cabin = pandas.get_dummies(titanic_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pandas.get_dummies(titanic_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pandas.get_dummies(titanic_train['Sex'], prefix= 'Sex')

dummies_Pclass = pandas.get_dummies(titanic_train['Pclass'], prefix= 'Pclass')
print dummies_Cabin
#precessed_train = pandas.concat([titanic_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
precessed_train = pandas.concat([titanic_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
precessed_train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
precessed_train["Age"]=precessed_train["Age"].fillna(precessed_train["Age"].median())
precessed_train["Fare"]=precessed_train["Fare"].fillna(precessed_train["Fare"].median())

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(precessed_train['Age'])
precessed_train['Age_scaled'] = scaler.fit_transform(precessed_train['Age'], age_scale_param)
fare_scale_param = scaler.fit(precessed_train['Fare'])
precessed_train['Fare_scaled'] = scaler.fit_transform(precessed_train['Fare'], fare_scale_param)
precessed_train.drop(['Age', 'Fare'], axis=1, inplace=True)

#test precess
dummies_Cabin = pandas.get_dummies(titanic_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pandas.get_dummies(titanic_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pandas.get_dummies(titanic_test['Sex'], prefix= 'Sex')
dummies_Pclass = pandas.get_dummies(titanic_test['Pclass'], prefix= 'Pclass')

#precessed_test = pandas.concat([titanic_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
precessed_test = pandas.concat([titanic_test, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
precessed_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
precessed_test["Age"]=precessed_test["Age"].fillna(precessed_test["Age"].median())
precessed_test["Fare"]=precessed_test["Fare"].fillna(precessed_test["Fare"].median())

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(precessed_test['Age'])
precessed_test['Age_scaled'] = scaler.fit_transform(precessed_test['Age'], age_scale_param)
fare_scale_param = scaler.fit(precessed_test['Fare'])
precessed_test['Fare_scaled'] = scaler.fit_transform(precessed_test['Fare'], fare_scale_param)
precessed_test.drop(['Age', 'Fare'], axis=1, inplace=True)

#print precessed_train.head()
#predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#alg=LinearRegression()
alg=RandomForestRegressor()
kf=KFold(precessed_train.shape[0],n_folds=2,random_state=1)

predicters=[t for t in precessed_train][2:]
predictions=[]
for train,test in kf:
    train_predictors=(precessed_train[predicters].iloc[train,:])
    train_target=precessed_train["Survived"].iloc[train]
    alg.fit(train_predictors,train_target)
    #print titanic_train[predictors].iloc[test,:]
    test_predictions=alg.predict(precessed_train[predicters].iloc[test,:])
    predictions.append(test_predictions)


predictions=np.concatenate(predictions,axis=0)
predictions[predictions>0.5]=1
predictions[predictions<=0.5]=0
accuacy=sum(predictions[predictions==precessed_train['Survived']])/len(predictions)

print accuacy
# print precessed_train.shape
# print precessed_test.shape
# print precessed_train.head()
# print precessed_test.head()
#print titanic_test[predictors].iloc[:,:]
predictions_result=alg.predict(precessed_test[predicters].iloc[:,:])
predictions_result[predictions_result>0.5]=1
predictions_result[predictions_result<=0.5]=0
out_result = pandas.DataFrame({'PassengerId':titanic_test['PassengerId'].as_matrix(), 'Survived':predictions_result.astype(np.int32)})
out_result.to_csv("data/logistic_regression_predictions2.csv", index=False)
#print predictions_result

