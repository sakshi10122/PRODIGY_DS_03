#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df= pd.read_csv('C:/Users/Rajesh Gonnade/Downloads/bank.csv')
print(df.head())


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


data=pd.DataFrame(df)
cols=data.select_dtypes("object").columns
cols


# In[9]:


for var in df.columns:
    plt.hist(df[var], bins=30)
    plt.title(var)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.show()
    

    


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(x ='job', data=df, hue="y")
plt.title('Distribution of job')
plt.xlabel('Job')
plt.ylabel('Count')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.show()


# In[19]:


corr_matrix = df.corr()
plt.figure(figsize=(17, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[20]:


LE=LabelEncoder()
df[cols]=df[cols].apply(LE.fit_transform)


# In[21]:


df.head()


# In[22]:


#MODEL BUILDING DECISION TREE

features = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
x=df[features]
y=df['y']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
dtree= DecisionTreeClassifier()

# Fit the model on the training data
dtree=dtree.fit(x_train, y_train)

# Make predictions on the test data
y_pred = dtree.predict(x_test)


# Plot the decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(dtree,feature_names=features,class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()


# In[23]:


dtree.score(x_test,y_test)
print(classification_report(y_test,y_pred))


# In[24]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[25]:


from sklearn.model_selection import cross_val_score
# Perform cross-validation (e.g., with 5 folds)
cv_scores = cross_val_score(dtree, x, y, cv=5, scoring='accuracy')


# In[26]:


# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy: {:.4f}".format(np.mean(cv_scores)))


# In[ ]:




