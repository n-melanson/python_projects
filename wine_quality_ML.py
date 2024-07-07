#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import math as math
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

#import files for analysis

file = "C:/Users/21mel/OneDrive/Documents/datasets/wine+quality/winequality-red.csv"
wine_df = pd.read_csv(file, delimiter=';')

print(wine_df.head(1))

#perform some basic analysis on the table
wine_df.info()
#sort and collect unique values and unique value counts
for column in wine_df.columns:
    unique_values = sorted(wine_df[column].unique())
    print('ColumnName: {}, Unique Values: {}'.format(column, unique_values))
print('\n')
#unique column values    
for column in wine_df.columns:
    print('ColumnName: {}, Count of Unique Values: {}'.format(column, wine_df[column].nunique()))
#total number of rows
print('\nNumber of Rows: ' + str(len(wine_df)))

#count occurances of each quality score
quality_score_chart = wine_df['quality'].value_counts().sort_index()

#bar chart of quality values
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
quality_score_chart.plot(kind='bar', ax=ax, color='green')

#create labels & titles
ax.set_title('Proportion of Quality Scores', 
             fontsize=18)
ax.set_ylabel('Proportion', 
              fontsize=14)
ax.set_xlabel('Quality Score', 
              fontsize=14)
ax.tick_params(rotation='auto')

plt.show()

#split data
X = wine_df.drop(columns='quality')
y = wine_df['quality']

#pull in standard scaler, scale our X data
sk_scaler = StandardScaler()

X_scaled = sk_scaler.fit_transform(X)
X_scaled_wine_df = pd.DataFrame(X_scaled, columns=X.columns)

#change scale of y data to go from 0-5 instead of 3-8
y_scaled = y.replace([3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5])

X_train, X_test, y_train, y_test = train_test_split(X_scaled_wine_df, y_scaled, test_size=0.25, random_state=42)

print("Scaled and Split Data:")
print(X_train.head())

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

#examine and test different models
models = {
    #attempt 1
    'Random Forest': RandomForestClassifier(random_state = 42, class_weight = class_weights_dict),
    'Gradient Boosting': GradientBoostingClassifier(random_state = 42),
    'XGBoost': XGBClassifier(random_state = 42),
    #adding more models, attempt 2. best accuracy score so far is .68 from XGBoost model.
    'SVM': SVC(random_state=42, class_weight='balanced'),
    'k-NN': KNeighborsClassifier()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred, zero_division = 0))
    print(confusion_matrix(y_test, y_pred))
    print('---------------------------------')
    
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight=class_weights_dict),
                           param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Random Forest Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

