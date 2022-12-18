#!/usr/bin/env python
# coding: utf-8

# #                           Predicting breast cancer in a patient

# Observation: Based on the features and parameters given, our data model should perform whether the patient is having Breast Cancer or not.Our model should have more accuracy and higher performance metrics.

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


#Ignore Warnings
from warnings import filterwarnings
filterwarnings('ignore')
# rcParams
plt.rcParams['figure.figsize']=[10,5]
#'exponential' to float
np.set_printoptions(suppress = True)
pd.options.display.float_format = '{:.2f}'.format


# # Load dataset

# In[3]:


dataframe = pd.read_csv("cancer.csv")


# In[4]:


data = dataframe.copy(deep = True)  #copy of data so that changes will not affect the source data


# In[5]:


data.head()


# In[ ]:


Observation: Our data set contains 569 rows and 33 columns.

IndependentVariable(Predictorvariables):radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concavepoints_mean,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst

Dependent Variable(Target Variable): Diagnosis

The data is labeled whereas gives input and expected output, so we go with Supervised Learning Algorithm.

Data is in the form of Categorical representation that is the output values cannot be compared instead they are given as M or B.

Therefore we go with Classification Model.


# # Data Pre-processing

# In[ ]:


#Steps involved for Data Processing
# 1. Data type, dimension
# 2. missing data
#3. data correction
#4. Statistical summary


# # 1)Data type

# In[6]:


data.info()


# Observation: Our data set consists of 31 Float values, 1 int value and 1 Object value. 

# In[7]:


data.shape


# In[8]:


dataframe.shape


# Checking if the data is copied correctly

# # 2) Missing data

# In[9]:


# to check the missing value
missing_values = dataframe.isnull().sum()
print(missing_values)


# In[10]:


data.drop("Unnamed: 32", axis = 1, inplace = True) #The Unnamed: 32 column contain null values and so the whole column is deleted


# In[11]:


data.drop("id", axis = 1, inplace = True)


# In[12]:


data


# # 3)Data Correction

# In[13]:


data.dtypes


# Observation:The 'diagnosis' column contains 'object' datatype and it is a categorical data which is invalid as system can only understand numeric data. 

# In[14]:


data['diagnosis'].unique()


# Check the unique values present in the column: 'diagnosis'. It contains 'M'and 'B' which can be converted to numeric values representing 0 and 1

# In[15]:


diagnosis = pd.get_dummies(data['diagnosis'],drop_first=True)


# In[16]:


diagnosis


# In[17]:


data = pd.concat([data,diagnosis],axis=1)  #appended at the end if axis is 1 denotes columns; axis is 0 denotes rows


# In[18]:


data


# In[19]:


data.columns


# # Detecting Outliers

# In[39]:


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum() 


# In[40]:


data = data.drop_duplicates()


# cap the outlier in the training data, as the testing data may or maynot have the outliers

# In[41]:


def capping(data,cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        Up=Q3 + (1.5 * IQR)
        Low=Q1 - (1.5 * IQR)
        
        data[col]=np.where(data[col]> Up,Up,np.where(data[col]<Low,Low,data[col]))


# In[42]:


cols=x_train.columns
cols


# In[43]:


x_train.describe()


# In[44]:


Q1 = x_train.quantile(0.25)
Q3 = x_train.quantile(0.75)
IQR = Q3 - Q1
((x_train < (Q1 - 1.5 * IQR)) | (x_train > (Q3 + 1.5 * IQR))).sum()


# # 4)Statistical description of dataframe 

# In[49]:


data.describe()


# In[50]:


#for categorical summary 
data.describe(include = object) 


# # Exploratory Data Analysis

# In[51]:


fig= data.hist(figsize = (18,18))


# In[52]:


# Univariate Analysis
data.radius_mean.describe()


# In[53]:


sns.distplot(data.radius_mean)


# In[54]:


plt.figure(figsize = (20,7))
stats.probplot(data['radius_mean'], plot = plt)
plt.show


# In[55]:


print("Skewness: %f" % data['radius_mean'].skew())


# In[56]:


# Multivariate Analysis
# seaborn pairplot is used to plot pairwise relationships between variables within a dataset
sns.pairplot(data.iloc[:,0:5],hue ='diagnosis')


# # Feature Engineering

# In[57]:


data.shape


# In[58]:


# Heat map allows us to check the correlation of the dataset like how they are correated with two or more continues variables
plt.figure(figsize= (25,10))
sns.heatmap(data.corr(),  annot = True,cmap= "cubehelix_r")
plt.show()


# In[59]:


# Covariance
plt.figure(figsize = (30,20))
sns.heatmap(data.cov(), annot = True, linewidth = 0.5)
plt.show()


# # Classifier

# Let us apply all the possible classification algorithmns and check which is giving the best accuracy for the given dataset

# In[20]:


# 1) Logistic Regression

x = data.drop(['diagnosis'],axis=1)
y = data[['diagnosis']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
log = LogisticRegression()
log.fit(x_train,y_train)
predicted = log.predict(x_test)
print("The Accuracy of Logistic Regression is: ",accuracy_score(y_test,predicted))


# In[21]:


print("The Classification Report of Logistic Regression is: ",classification_report(y_test,predicted))


# In[22]:


# 2) KNN

from sklearn.neighbors import KNeighborsClassifier
x = data.drop(['diagnosis'],axis=1)
y = data[['diagnosis']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
regg = KNeighborsClassifier()
regg.fit(x_train,y_train)
predicted = regg.predict(x_test)
print("The Accuracy of KNN is: ",accuracy_score(y_test,predicted))


# In[23]:


print(confusion_matrix(y_test,predicted))


# In[24]:


print("The Classification Report of KNN is: ",classification_report(y_test,predicted))


# In[85]:


# 3)SVM
from sklearn.svm import SVC
x = data.drop(['diagnosis'],axis=1)
y = data[['diagnosis']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
sv = SVC()
sv.fit(x_train,y_train)
predicted = sv.predict(x_test)
print("The Accuracy of SVM is: ",accuracy_score(y_test,predicted))


# In[86]:


print("The Classification Report of SVM is: ",classification_report(y_test,predicted))


# In[27]:


# 4)Decision Trees
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
x = data.drop(['diagnosis'],axis=1)
y = data[['diagnosis']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
obj = DecisionTreeClassifier()
dtree = obj.fit(x_train,y_train)
print("The Accuracy of Decision Tree is: ",accuracy_score(y_test,predicted))


# In[28]:


print("The Classification Report of Decision Tree is: ",classification_report(y_test,predicted))


# In[29]:


print(confusion_matrix(y_test,predicted))


# Observations: Out of 4 classification algorithms applied, we got highest accuracy in SVM algorithm. So lets apply SVM model for our dataset.

# # Feature Analysis

# In[30]:


from sklearn.preprocessing import StandardScaler
feature_scaling = StandardScaler()
x_train= feature_scaling.fit_transform(x_train)
x_test = feature_scaling.fit_transform(x_test)
x_train


# # Classification Report

# In[31]:


#Test the model using SVM
SVM = SVC()
SVM.fit(x_train,y_train)
predicted_xtest = SVM.predict(x_test)
predicted_xtrain = SVM.predict(x_train)
print(classification_report(y_test,predicted_xtest))


# # Accuracy of Training Data

# In[32]:


print(accuracy_score(y_train,predicted_xtrain))


# # Accuracy of Testing Data

# In[33]:


print(accuracy_score(y_test,predicted_xtest))


# Observation: In both training and testing data we got accuracy as 1.0, therefore we got 100% accuracy in the SVM model.
# This concludes that both training and testing data are perfectly fitted.

# In[71]:


from sklearn.preprocessing import StandardScaler

std = StandardScaler()

x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)


# In[72]:


#from sklearn import svm
from sklearn.svm import SVC

SVM = SVC(kernel='linear', gamma='scale')
SVM.fit(x_train, y_train) 


# # Predicting the test test results

# In[73]:


y_pred = SVM.predict(x_test)
y_pred


# # Confusion Metrics

# In[75]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 


# In[76]:


cnf_matrix_test = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_test, display_labels=SVM.classes_).plot()


# In[77]:


from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("The model accuracy is", accuracy )


# In[87]:


group_names = ["True Pos","False Pos","False Neg","True Neg"]
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')


# In[78]:


from sklearn.metrics import classification_report
predictions = SVM.predict(x_test)
print(classification_report(y_test, predictions))


# # Conclusion: We got 100% accuracy in both training and testing data using SVM model. Hence we can say that our data set is perfectly fitted

# In[ ]:




