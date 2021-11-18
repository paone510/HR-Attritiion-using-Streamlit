# importing the dataset
import pandas as pd
 
data = pd.read_csv('C:/Users/Hp/Downloads/Final dataset Attrition.xlsx - Sheet1.csv')   

#Remove the column EmployeeNumber
data = data.drop(['date of termination','Status of leaving','absenteeism','Work_accident','Source of Hire','Jobmode','HigherEducation','Date of Hire','modeofwork','leaves','Gender'], axis = 1) # A number assignment 

data_copy=data.copy()
target = 'Attrition'
category_col =[ 'BusinessTravel', 'Department', 'JobRole', 'MaritalStatus','OverTime',]

for col in category_col:
    dummy = pd.get_dummies(data_copy[col], prefix=col)
    data_copy = pd.concat([data_copy,dummy], axis=1)
    del data_copy[col]

target_mapper = {'Yes':0, 'No':1}
def target_encode(val):
    return target_mapper[val]

data_copy['Attrition'] = data_copy['Attrition'].apply(target_encode)

#Split the data into independent 'X' and dependent 'Y' variables
X = data_copy.drop('Attrition',axis=1)
Y = data_copy['Attrition']

# Split the dataset into 75% Training set and 25% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#Use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X, Y)


# Saving the model
import pickle
pickle.dump(forest, open('attrition_clf.pkl', 'wb'))
