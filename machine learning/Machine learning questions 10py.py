#!/usr/bin/env python
# coding: utf-8

# # Machine learning 

# ### Q10:- An Ad- Agency analyzed a dataset of online ads and used a machine learningmodel to predict whether a user would click on an ad or not.
# Dataset This is the Dataset You can use this dataset for this question.

# In[ ]:


#solution:-

## Import the required libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## Load the dataset into a Pandas DataFrame:-
data = pd.read_csv('Downloads/train.csv/train.csv')


## Data preprocessing:
### Remove unnecessary columns:

data.drop(['id', 'hour', 'device_id', 'device_ip'], axis=1, inplace=True)

### Encode categorical features:

label_encoders = {}
categorical_features = ['site_id', 'site_domain', 'app_id', 'app_domain', 'device_model']

for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    data[feature] = label_encoders[feature].fit_transform(data[feature].astype(str))


## Split the data into training and testing sets:
X = data.drop('click', axis=1)
y = data['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Train a Random Forest classifier:
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# In[ ]:


import dask.dataframe as dd
import mlflow
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score

# Read the dataset using Dask
df = dd.read_csv('train.csv')

# Split the data into features (X) and target (y)
X = df.drop('friend_request_accepted', axis=1)
y = df['friend_request_accepted']

# Split the data into training and testing sets using Dask's train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log the metrics using MLflow
with mlflow.start_run():
    mlflow.log_param('model', 'Logistic Regression')
    mlflow.log_metric('accuracy', accuracy)

