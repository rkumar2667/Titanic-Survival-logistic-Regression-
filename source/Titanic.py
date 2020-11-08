#!/usr/bin/env python
# coding: utf-8

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')

#Read the data & clean it
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Combining the data for cleaning and estimation of missing values
combined = pd.concat([ train, test ])
combined.describe()


#Finding the missing data
sns.heatmap(combined.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#filling the missing age data by mean values

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]   
          
    if pd.isnull(Age): 
        
        if Pclass == 1:            
            return np.mean(combined[combined['Pclass'] == 1 ]['Age'])

        elif Pclass == 2:
            return np.mean(combined[combined['Pclass'] == 2 ]['Age'])

        else:
            return np.mean(combined[combined['Pclass'] == 3 ]['Age'])

    else:
        return Age
    

combined['Age'] = combined[['Age','Pclass']].apply(impute_age, axis=1)   


#dropping the cabin column
combined.drop('Cabin', axis=1, inplace=True)
#Fill the row in Embarked that is NaN with 'S' (most common port), and the row in Fare with mean
combined.fillna(value={'Embarked': 'S', 'Fare': np.mean(combined['Fare'])}, inplace=True)


#Converting categorical features to dummy variables using pandas
combined.info()

sex = pd.get_dummies(combined['Sex'], drop_first=True)
embark = pd.get_dummies(combined['Embarked'], drop_first=True)
combined.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
combined = pd.concat([combined,sex,embark], axis=1)
combined.head()

#Building the model

train = combined[combined['Survived'].notnull()]
test = combined[combined['Survived'].isnull()]
test = test.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'],axis=1), 
                                                    train['Survived'], test_size=0.30)

#Fitting into model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test).astype(int)



#Evaluating the model
print(classification_report(y_test,predictions))

