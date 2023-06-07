#!/usr/bin/env python
# coding: utf-8

# # Advanced Machine Learning

# ### Q- 3. A company wants to predict the sales of its product based on the money spent on different platforms for marketing. They want you to figure out how they can spend money on marketing in the future in such a way that they can increase their profit as much as possible built-in docker and use some library to display that in frontend
# Dataset This is the Dataset You can use this dataset for this question. Note:
# Use only Dask

# In[ ]:


# Import the required libraries
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.regressor import LinearRegression
import matplotlib.pyplot as plt

#Load the dataset using Dask
df = dd.read_csv("marketing.csv")

# Prepare the data for modeling
X = df.drop("sales", axis=1)
y = df["sales"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Visualize the predicted sales:
# Convert Dask dataframe to Pandas dataframe for visualization
X_test_pd = X_test.compute()
y_test_pd = y_test.compute()

# Plot the actual vs predicted sales
plt.scatter(X_test_pd["TV"], y_test_pd, color="blue", label="Actual")
plt.scatter(X_test_pd["TV"], y_pred.compute(), color="red", label="Predicted")
plt.xlabel("Money spent on TV marketing")
plt.ylabel("Sales")
plt.legend()
plt.show()


# **Note-** `To optimize future marketing spending, you can use techniques such as budget allocation optimization, which involves finding the optimal allocation of the marketing budget across different platforms.`
# 
# `While Dask is a powerful tool for distributed computing and handling large datasets, it does not provide built-in frontend capabilities. To display the frontend with interactive plots and visualizations, you may need to use additional libraries or frameworks such as Flask, Dash, or Streamlit.`

# In[ ]:




