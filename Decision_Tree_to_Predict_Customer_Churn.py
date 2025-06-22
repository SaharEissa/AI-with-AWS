import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

warnings.filterwarnings('ignore')

# Creating a synthetic dataset
# This dataset simulates customer data for a telecom company

data = {
      'CustomerID': range(1, 101),  # Unique ID for each customer
      'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]*10,  # Age of customers
      'MonthlyCharge': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]*10,  # Monthly bill amount
      'CustomerServiceCalls': [1, 2, 3, 4, 0, 1, 2, 3, 4, 0]*10,  # Number of customer service calls
      'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']*10  # Churn status
}

df = pd.DataFrame(data)
# print(df.head(5))

# Now it's time to split the data into features and target variables
# Features include age, monthly charge, and customer Service Calls
# The Target Variable is Churn  (Yes, No)

featuresX =df[['Age', 'MonthlyCharge','CustomerServiceCalls'] ]
targetY = df['Churn']
# print(featuresX)
# print(targetY)

# After defining the data into features and target, we need to split the data into training and testing
# Usually 70% of the data is used for training and 30% for testing
# Both features and target split into training and testing

featuresX_train, featuresX_test, targetY_train, targetY_test = train_test_split(featuresX, targetY, test_size=0.3, random_state=42)

# Training the Decision Tree Model
clf =DecisionTreeClassifier()
clf.fit(featuresX_train, targetY_train)

# Making predictions on the test set
targetY_pred = clf.predict(featuresX_test)

# Evaluating the model using accuracy
# Accuracy is the proportion of correct predictions among the total number of cases processed
accuracy = accuracy_score(targetY_test, targetY_pred)
print(f'Model Accuracy: {accuracy}')

# Visualizing the decision tree
# This visualization helps to understand how the model makes decisions
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=['Age', 'MonthlyCharge', 'CustomerServiceCalls'], class_names=['No Churn', 'Churn'])
plt.title('Decision Tree For Predicting Customer Churn')
plt.show()