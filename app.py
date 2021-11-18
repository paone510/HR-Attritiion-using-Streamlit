import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.sidebar.header('User Input Features')

st.sidebar.markdown('Attrition Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        BusinessTravel = st.sidebar.selectbox('Business Travel',('Travel_Rarely','Travel_Frequently','Non-Travel'))
        Department = st.sidebar.selectbox('Department',('Research & Development','Human Resources','Sales'))
        JobRole = st.sidebar.selectbox('Job Role',('Sales Executive','Research Scientist','Laboratory Technician','Manufacturing Director','Healthcare Representative','Manager','Sales Representative','Research Director','Human Resources'))
        MaritalStatus = st.sidebar.selectbox('Marital Status',('Married','Single','Divorced'))
        OverTime = st.sidebar.selectbox('Over Time',('Yes','No'))
        Age = st.sidebar.slider('Age', 18,60,45)
        DistanceFromHome = st.sidebar.slider('Distance From Home', 1,30,12)
        JobInvolvement = st.sidebar.slider('Job Involvement', 1,4,2)
        JobLevel = st.sidebar.slider('Job Level', 1,5,3)
        JobSatisfaction = st.sidebar.slider('Job Satisfaction', 1,4,2)
        MonthlyIncome = st.sidebar.slider('Monthly Income', 1000,20000,6000)
        NumCompaniesWorked = st.sidebar.slider('Num Companies Worked', 0,9,5)
        PercentSalaryHike = st.sidebar.slider('Percent Salary Hike', 0,50,25)
        PerformanceRating = st.sidebar.slider('Performance Rating', 1,4,2)
        StockOptionLevel = st.sidebar.slider('Stock Option Level', 0,3,1)
        TotalWorkingYears = st.sidebar.slider('Total Working Years', 0,40,20)
        TrainingTimesLastYear = st.sidebar.slider('Training Times Last Year', 1,6,3)
        YearsAtCompany = st.sidebar.slider('YearsAtCompany', 0,40,20)
        YearsSinceLastPromotion = st.sidebar.slider('YearsSinceLastPromotion', 0,15,7)
        YearsWithCurrManager = st.sidebar.slider('YearsWithCurrManager', 0,17,8)
        
        data = {'Age': Age,
                'BusinessTravel': BusinessTravel,
                'Department': Department,
                'DistanceFromHome': DistanceFromHome,
                'JobInvolvement': JobInvolvement,
                'JobLevel': JobLevel,
                'JobRole': JobRole,
                'JobSatisfaction': JobSatisfaction,
                'MaritalStatus': MaritalStatus,
                'MonthlyIncome': MonthlyIncome,
                'NumCompaniesWorked': NumCompaniesWorked,
                'OverTime': OverTime,
                'PercentSalaryHike': PercentSalaryHike,
                'PerformanceRating': PerformanceRating,
                'StockOptionLevel': StockOptionLevel,
                'TotalWorkingYears': TotalWorkingYears,
                'TrainingTimesLastYear': TrainingTimesLastYear,
                'YearsAtCompany': YearsAtCompany,
                'YearsSinceLastPromotion': YearsSinceLastPromotion,
                'YearsWithCurrManager': YearsWithCurrManager,}
        features = pd.DataFrame(data, index=[0])
        return features
    input_data = user_input_features()
    

# Combines user input features with entire attrition dataset
# This will be useful for the encoding phase
data_raw = pd.read_csv('C:/Users/Hp/Downloads/Final dataset Attrition.xlsx - Sheet1.csv')
#Remove the column EmployeeNumber
data_raw = data_raw.drop(['date of termination','Status of leaving','absenteeism','Work_accident','Source of Hire','Jobmode','HigherEducation','Date of Hire','modeofwork','leaves','Gender'], axis = 1) # A number assignment 

data = data_raw.drop(columns=['Attrition'])
df = pd.concat([input_data,data],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
category_col =[ 'BusinessTravel', 'Department', 'JobRole', 'MaritalStatus','OverTime',]
for col in category_col:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('attrition_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
Attrition_prediction = np.array(['Yes','No'])
st.write(Attrition_prediction[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)