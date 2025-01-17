#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import math as math
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import *

#bring in file location and create data frame
file_repository = "C:/Users/21mel/OneDrive/Documents/datasets/Telco_Customer_Churn_Kaggle/Telco-Customer-Churn.csv"
column_names = ["customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBuilding","PaymentMethod","MonthlyCharges","TotalCharges","Churn"]
telco_data = pd.read_csv(file_repository, delimiter = ',', header = 0)

#take a look at how the data looks
print(telco_data.head(1))

#perform exploratory analysis on data & metadata

for column in telco_data.columns:
    print('Column Name: {}; Unique Values: {}'.format(column, telco_data[column].unique()))
#basic metadata
telco_data.info()
#change datatype for TotalCharges to numeric from object
telco_data['TotalCharges'] = pd.to_numeric(telco_data['TotalCharges'], errors = 'coerce')
print('TotalCharges datatype to numeric confirmed.')
#check TotalCharges 
telco_data[telco_data['TotalCharges'].isnull()]
#11 of the totalcharge values are null, they also all have a tenure of 0 and have not churned, so I will assume they are either in their first month of the contract or some error exists as monthly charges are NOT null -- removing from dataset.
telco_data.dropna(inplace=True)
print('rows w/ null values removed')
#remove parethetic values from payment types
telco_data['PaymentMethod'] = telco_data['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
telco_data.PaymentMethod.unique()
#removed

#dive a little deeper with some basic visualizations
#bar chart of the Churn, let's see what proportion are yes vs no.
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)

prop_response = telco_data['Churn'].value_counts(normalize = True)

prop_response.plot(kind = 'bar',
                   ax=ax,
                   color=['green','red'])
ax.set_title('Proportion of customers that have churned',
             fontsize = 18)
ax.set_ylabel('Proportion of responses',
              fontsize = 14)
ax.set_xlabel('Response type - Yes or No',
              fontsize = 14)
ax.tick_params(rotation = 'auto')

#examine the impact of our demographic characteristics on the Churn variable
def percentage_stacked_plot(columns_to_plot, super_title):
        
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        prop_by_independent = pd.crosstab(telco_data[column], telco_data['Churn']).apply(lambda x: x/x.sum()*100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['green','red'])
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)
        ax.set_title('Proportion of observations by ' + column,
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
           
#Graph out the various groups        
# demographic column names
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
percentage_stacked_plot(demographic_columns, 'Demographic Information')
#account columns
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']
percentage_stacked_plot(account_columns, 'Customer Info')


def histogram_plot(columns_to_plot, super_title):
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)
    fig =plt.figure(figsize=(12, 5 * number_of_rows))
    fig.suptitle(super_title, fontsize=22, y=.95)
    
    for index, column in enumerate(columns_to_plot, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        telco_data[telco_data['Churn']=='No'][column].plot(kind='hist', ax=ax, density=True,
                                                           alpha=0.5, color='green', label='No')
        telco_data[telco_data['Churn']=='Yes'][column].plot(kind='hist', ax=ax, density=True,
                                                            alpha=0.5, color='red', label='Yes')
        ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)
        ax.set_title('Distribution of ' + column + ' by churn',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')
        
        #elim border
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
            
# customer account column names
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
# histogram of costumer account columns 
histogram_plot(account_columns_numeric, 'Customer Account Information')

#service type columns
services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
percentage_stacked_plot(services_columns, 'Services Information')

#examine relationships between variables
def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, telco_data.Churn)

categorical_variables = telco_data.select_dtypes(include=object).drop('Churn', axis = 1)
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

print(feature_importance)

#multipleLines, PhoneService, and gender all have very low correspondence with the target variable churn, dropping variables from dataset.
#telco_data.drop(columns=['gender','MultipleLines','PhoneService'])

#create new dataset, transform label data into binaries 
telco_data_transformed = telco_data.copy()

label_encoded_columns = ['gender','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for column in label_encoded_columns:
    if column == 'gender':
        telco_data_transformed[column] = telco_data[column].map({'Female':1, 'Male':0})
    else:
        telco_data_transformed[column] = telco_data[column].map({'Yes':1, 'No':0})

#normalize non-binary variables to binary format for each level
one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

telco_data_transformed = pd.get_dummies(telco_data_transformed, columns = one_hot_encoding_columns)

#remove cusotmerID

telco_data_transformed = telco_data_transformed.drop(columns = 'customerID')

#numeric variable normalization
min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

for column in min_max_columns:
    min_column = telco_data_transformed[column].min()
    max_column = telco_data_transformed[column].max()
    telco_data_transformed[column] = (telco_data_transformed[column] - min_column) / (max_column - min_column)
    
#split data for training and testing sets, begin with defining variable types
X = telco_data_transformed.drop(columns='Churn')
y = telco_data_transformed['Churn']

print(X.columns)
print(y.name)

#numpy arrays
X = X.to_numpy()
y = y.to_numpy()

#ensure contiguous data
X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)


#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=40, shuffle=True)

##create models
def create_models(seed=2):

    models = []
    models.append(('dummy_classifier', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('k_nearest_neighbors', KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('support_vector_machines', SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    
    return models

# create a list with all the algorithms we are going to assess
models = create_models()

#test each model
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    # fit the model with the training data
    model.fit(X_train, y_train).predict(X_test)
    # make predictions with the testing data
    predictions = model.predict(X_test)
    # calculate accuracy 
    accuracy = accuracy_score(y_test, predictions)
    # append the model name and the accuracy to the lists
    results.append(accuracy)
    names.append(name)
    # print classifier accuracy
    print('Classifier: {}, Accuracy: {}'.format(name, accuracy))
    
#check results - based on the scores, gradient boosting will be the best model for this test.    
grid_parameters = {'n_estimators': [80, 90, 100, 110, 115, 120],
                   'max_depth': [3, 4, 5, 6],
                   'max_features': [None, 'auto', 'sqrt', 'log2'], 
                   'min_samples_split': [2, 3, 4, 5]}


# define the RandomizedSearchCV class for trying different parameter combinations
random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=grid_parameters,
                                   cv=5,
                                   n_iter=150,
                                   n_jobs=-1)

# fitting the model for random search 
random_search.fit(X_train, y_train)

# print best parameter after tuning
print(random_search.best_params_)

#{'n_estimators': 90, 'min_samples_split': 4, 'max_features': 'sqrt', 'max_depth': 3}

# make the predictions
random_search_predictions = random_search.predict(X_test)

# construct the confusion matrix
confusion_matrix = confusion_matrix(y_test, random_search_predictions)

# visualize the confusion matrix
confusion_matrix


print(classification_report(y_test, random_search_predictions))
accuracy_score(y_test, random_search_predictions)


# In[99]:


grid_parameters = {'n_estimators': [80, 90, 100, 110, 115, 120],
                   'max_depth': [3, 4, 5, 6],
                   'max_features': [None, 'auto', 'sqrt', 'log2'], 
                   'min_samples_split': [2, 3, 4, 5]}


# define the RandomizedSearchCV class for trying different parameter combinations
random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=grid_parameters,
                                   cv=5,
                                   n_iter=150,
                                   n_jobs=-1)

# fitting the model for random search 
random_search.fit(X_train, y_train)

# print best parameter after tuning
print(random_search.best_params_)

