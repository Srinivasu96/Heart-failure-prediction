#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[3]:


#importing dataset
df = pd.read_csv('C:\\Users\\cheru\\OneDrive\\Desktop\\archive\\heart_failure_clinical_records_dataset.csv')
df.head(10)


# In[4]:


#check the shape of the data
df.shape


# In[5]:


df.columns


# In[6]:


# check basic information of dataset
df.info()


# In[7]:


# check statistical description of dataset
df.describe()


# In[8]:


# check any null values present in dataset or not
df.isnull().sum()


# In[9]:


# Check Number of Unique Values in dataset
df.nunique()


# In[10]:


#data visualization
sns.countplot(x='DEATH_EVENT',data=df)
plt.show()
df.DEATH_EVENT.value_counts()


# In[12]:


#Finding the correlation between different features
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[13]:


# Plot histograms of each parameter 
df.hist(figsize = (20, 20))
plt.show()


# In[14]:


sns.histplot(df['age'], kde= True)
plt.show()


# In[15]:


sns.countplot(df.sex)
plt.show()

#Classifying 0 as Female and 1 as Male
classes = {0:'Female', 1:'Male'}
print(df.sex.value_counts().rename(index = classes))


# In[16]:


ax=sns.barplot(x='high_blood_pressure',y='platelets',hue='DEATH_EVENT',data=df)
plt.legend(loc=9)
plt.show()


# In[17]:


ax=sns.barplot(x='sex',y='platelets',hue='DEATH_EVENT',data=df)
plt.legend(loc=9)
plt.show()


# In[18]:


sns.distplot(df['creatinine_phosphokinase'], kde= True)
plt.show()


# In[19]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['creatinine_phosphokinase'])

plt.subplot(1,2,2)
sns.boxplot(df['creatinine_phosphokinase'])
plt.show()


# In[20]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['ejection_fraction'])

plt.subplot(1,2,2)
sns.boxplot(df['ejection_fraction'])
plt.show()


# In[21]:


df[df["DEATH_EVENT"] == True]["ejection_fraction"].plot.kde(label="Death event")
df[df["DEATH_EVENT"] == False]["ejection_fraction"].plot.kde(label="Survived")
plt.legend()
plt.show()


# In[22]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['platelets'])

plt.subplot(1,2,2)
sns.boxplot(df['platelets'])
plt.show()


# In[23]:


df[df["DEATH_EVENT"] == True]["platelets"].plot.kde(label="Death event")
df[df["DEATH_EVENT"] == False]["platelets"].plot.kde(label="Survived")
plt.legend()
plt.show()


# In[24]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['serum_creatinine'])

plt.subplot(1,2,2)
sns.boxplot(df['serum_creatinine'])
plt.show()


# In[25]:


df[df["DEATH_EVENT"] == True]["serum_creatinine"].plot.kde(label="Death event")
df[df["DEATH_EVENT"] == False]["serum_creatinine"].plot.kde(label="Survived")
plt.legend()
plt.show()


# In[26]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['serum_sodium'])

plt.subplot(1,2,2)
sns.boxplot(df['serum_sodium'])
plt.show()


# In[27]:


df[df["DEATH_EVENT"] == True]["serum_sodium"].plot.kde(label="Death event")
df[df["DEATH_EVENT"] == False]["serum_sodium"].plot.kde(label="Survived")
plt.legend()
plt.show()


# In[28]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['time'])

plt.subplot(1,2,2)
sns.boxplot(df['time'])
plt.show()


# In[29]:


df[df["DEATH_EVENT"] == True]["time"].plot.kde(label="Death event")
df[df["DEATH_EVENT"] == False]["time"].plot.kde(label="Survived")
plt.legend()
plt.show()


# In[30]:


plt.figure(figsize=(15,7))
df.corr()['DEATH_EVENT'].sort_values(ascending=False).drop(['DEATH_EVENT']).plot(kind='bar', color='g')
plt.xlabel("Feature",fontsize=14)
plt.title("Correlation",fontsize=18)
plt.show()


# In[31]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(df['ejection_fraction'])
plt.title("Before outlier Removal")

#remove outliers
df = df.drop(df[df.ejection_fraction >=70].index)

plt.subplot(1,2,2)
sns.boxplot(df['ejection_fraction'])
plt.title('After Outlier Removal')
plt.show()


# In[32]:


df[df.platelets > 650000]


# In[33]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(df['platelets'])
plt.title("Before outlier Removal")

df = df.drop(df[df.platelets >=650000].index)

plt.subplot(1,2,2)
sns.boxplot(df['platelets'])
plt.title('After Outlier Removal')
plt.show()


# In[34]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(df['serum_sodium'])
plt.title("Before outlier Removal")

df = df.drop(df[df.serum_sodium < 120].index)

plt.subplot(1,2,2)
sns.boxplot(df['serum_sodium'])
plt.title('After Outlier Removal')
plt.show()


# In[35]:


from sklearn.ensemble import ExtraTreesClassifier
X = df.drop(columns='DEATH_EVENT')
y = df['DEATH_EVENT']


# In[36]:


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


# In[37]:


# In terms of percentage
feat_importances*100


# In[38]:


feat_importances.nlargest(6)


# In[39]:


sum(feat_importances.nlargest(6))


# In[40]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = df.iloc[:,[0,2,4,7,8,11]]
y = df[['DEATH_EVENT']]


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[42]:


sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[46]:


#Logistic Regression Model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)


#Score
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print('Accuracy Score: ', lr_accuracy*100)


#Confusion Matrix
lr_cm = confusion_matrix(y_test, lr_y_pred)
classes_names = ['False', 'True']
lr_ConfusionMatrix = pd.DataFrame(lr_cm, index=classes_names, columns=classes_names)

sns.heatmap(lr_ConfusionMatrix, annot=True, cbar=None, cmap="OrRd", fmt = 'g')
plt.title("Logistic Regression Confusion Matrix") 
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

print("Classification Report:-")
print(  classification_report(y_test, lr_y_pred))


# In[47]:


#Decision Tree Model
dtree = DecisionTreeClassifier(criterion='gini',max_depth=5,random_state=33) #criterion can be entropy
dtree.fit(X_train, y_train)
dtree_y_pred = dtree.predict(X_test)


#Score
#Score
dtree_accuracy = accuracy_score(y_test, dtree_y_pred)
print('Accuracy Score: ', dtree_accuracy*100)

#Confusion Matrix
dtree_cm = confusion_matrix(y_test, dtree_y_pred)
dtree_ConfusionMatrix = pd.DataFrame(dtree_cm, index=classes_names, columns=classes_names)

sns.heatmap(dtree_ConfusionMatrix, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

print("Classification Report:-")
print(  classification_report(y_test, dtree_y_pred))


# In[48]:


#Random Forest Model
rfc = RandomForestClassifier(criterion = 'gini',n_estimators=200,max_depth=5,random_state=33, n_jobs=-1)
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)


#Score
#Score
rfc_accuracy = accuracy_score(y_test, rfc_y_pred)
print('Accuracy Score: ', rfc_accuracy*100)


#Confusion Matrix
rfc_cm = confusion_matrix(y_test, rfc_y_pred)
rfc_ConfusionMatrix = pd.DataFrame(rfc_cm, index=classes_names, columns=classes_names)

sns.heatmap(rfc_cm, annot=True, cbar=None, cmap="Greens", fmt = 'g')
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

print("Classification Report:-")
print(  classification_report(y_test, rfc_y_pred))


# In[49]:


#SVC Model
svc = SVC(kernel= 'rbf', max_iter=100, C=1.0, gamma='auto')
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)

#Score
#Score
svc_accuracy = accuracy_score(y_test, svc_y_pred)
print('Accuracy Score: ', svc_accuracy*100)


#Confusion Matrix
svc_cm = confusion_matrix(y_test, svc_y_pred)
svc_ConfusionMatrix = pd.DataFrame(svc_cm, index=classes_names, columns=classes_names)

sns.heatmap(svc_ConfusionMatrix, annot=True, cbar=None, cmap="Purples", fmt = 'g')

plt.title("SVC Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

print("Classification Report:-")
print(  classification_report(y_test, svc_y_pred))


# In[51]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)


#Score
#Score
gnb_accuracy = accuracy_score(y_test, gnb_y_pred)
print('Accuracy Score: ', gnb_accuracy*100)

#Confusion Matrix
gnb_cm = confusion_matrix(y_test, gnb_y_pred)
classes_names = ['Not Fraud', 'Fraud']
gnb_ConfusionMatrix = pd.DataFrame(gnb_cm, index=classes_names, columns=classes_names)

sns.heatmap(gnb_ConfusionMatrix, annot=True, cbar=None, cmap="Greens", fmt = 'g')
plt.title("Naive Bayes Classifier Confusion Matrix") 
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

print("Classification Report:-")
print(  classification_report(y_test, gnb_y_pred))

