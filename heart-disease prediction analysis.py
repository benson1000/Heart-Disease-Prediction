#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings("ignore")
from pandas.plotting import scatter_matrix


# In[2]:


plt.style.use("ggplot")


# In[3]:


data=pd.read_csv("heart_2020_cleaned.csv")


# In[4]:


data   #the dataset to use.


# In[5]:


data.head() # displays the first five rows of the data


# In[6]:


data.tail()  #displays the last five rows of the data


# In[7]:


data.shape  #the shape of the dataset, the train dataset and the test dataset.


# In[8]:


data.dtypes.value_counts()  #it returns the number of occurances of each datatype in the dataset.


# In[9]:


data.columns   #it displays the columns in the dataset.


# In[10]:


data.nunique()   #the number of unique values per column in the entire dataset


# In[11]:


data.info() #it provides information about the dataset. This information includes the Non-missing values per column, the columns datatype and the columns


# In[12]:


data.size  #the size of the file


# In[13]:


data.describe().T.style.background_gradient(cmap='viridis') #the descriptive statistics of the data. It works only on Numerical


# In[14]:


data.isnull().sum()  #the sum of missing values per column... There are no missing values in the dataset.


# In[15]:


plt.figure(figsize=(12,10))
msno.bar(data)  #visualizing the missing values
plt.show()


# # DATA CLEANING

# 1.DEALING WITH DUPLICATES

# In[16]:


duplicates=data[data.duplicated()]   #returns a dataframe of duplicates
duplicates.head()  #displays the first five duplicates in the duplicates dataframe


# In[17]:


duplicates.shape   #the shape of duplicates in duplicates dataframe. The original file has 18,078 rows of duplicates.


# In[18]:


data.drop_duplicates(inplace=True)   #dropping the duplicates from the original dataframe


# In[19]:


data.shape   #the new dataset that is free from duplicates.


# 2.OUTLIER DETECTION

# In[20]:


plt.figure(figsize=(12,10))
sns.boxplot(data=data,x="Smoking",y="BMI",hue="HeartDisease")
plt.show()


# Using INTERQUARTILE RANGE to detect OUTLIERS 
# 
# INTERQUARTILE RANGE (IQR) = q3(0.75) - q1(0.25)

# In[21]:


q1=data.quantile(0.25)   #the first quantile
q3=data.quantile(0.75)  #the third quantile
iqr=q3-q1
iqr


# #getting the actual values that are outliers in our dataset

# In[22]:


(data < (q1 -1.5 *iqr)) | (data > (q3 + 1.5 * iqr))


# The data point that is True is an outlier in the dataset

# In[23]:


outliers=data[(data < (q1 -1.5 * iqr)) | (data > (q3 + 1.5 * iqr))].any(axis=1)
data[outliers].head()


# In[24]:


outliers_df=data[outliers]  #the outliers in our dataset.
outliers_df.shape   #79186 rows of data are outliers


# In[25]:


scatter_matrix(outliers_df,figsize=(12,12),color='green')  #visualizing the outliers dataframe
plt.show() 


# selecting the dataset that has no outliers

# In[26]:


new_data=data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)]


# In[27]:


data=new_data
data.set_index(np.arange(0,222531),inplace=True)   #setting the index of the dataframe


# In[28]:


data.shape  #the shape of the dataset without outliers


# Finding the Correlation between Numerical Variables

# In[29]:


plt.figure(figsize=(16,8))
sns.heatmap(data.corr(),annot=True,fmt='0.1%',cmap='viridis')
plt.show()


# In[30]:


plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
plot1=sns.countplot(data=data,x="HeartDisease")
plt.title("Heart Disease Distribution",color="white",backgroundcolor="black",fontsize=20)
plot1.bar_label(plot1.containers[0],color='black',size=17),plt.yticks(()),plt.ylabel(None)

plt.subplot(1,2,2)
labels=data['HeartDisease'].value_counts().index
vals=data['HeartDisease'].value_counts().values
plt.pie(vals,autopct="%1.2f%%",colors=['green','blue'],frame=True,shadow=True,wedgeprops={'edgecolor':'k'},
       textprops={'color':'white','size':22,'backgroundcolor':'black'}),plt.tight_layout(),plt.axis("equal"),plt.yticks(())
plt.xticks(()), plt.legend(labels,fontsize=15),plt.title("Heart Disease Distribution",color="white",backgroundcolor="black",fontsize=20)
plt.show()


# In[31]:


data


# In[32]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
ax1=sns.countplot(data=data,x="Sex")
ax1.bar_label(ax1.containers[0],size=17),plt.yticks(()),plt.ylabel(None)
plt.title(' Sex',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')

plt.subplot(1,2,2)
ax2=sns.countplot(data=data,x="Sex",hue="HeartDisease")
ax2.bar_label(ax2.containers[0],size=16) , ax2.bar_label(ax2.containers[1],size=16,color='green')
plt.title('Sex by HeartDisease',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')
plt.yticks(()),plt.ylabel(None)
plt.show()


# From the above case, it can be said that men are more likely to get heart Disease problems than women

# In[33]:


plt.figure(figsize=(18,18))
plt.subplot(2,2,1)
color=['seagreen','#34495E', 'brown','violet']
ax3 = plt.bar(data['Diabetic'].value_counts().index,data['Diabetic'].value_counts().values,label='Diabetic',color=color)
plt.bar_label(ax3,size=17,color='black'),plt.yticks(()),plt.ylabel(None)
plt.title('Diabetic',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')

plt.subplot(2,2,2)
ax4=sns.countplot(data=data,x='Diabetic',hue='HeartDisease',palette='winter_r')
ax4.bar_label(ax4.containers[0],size=12,color='k'),ax4.bar_label(ax4.containers[1],size=12,color='red')
plt.title('Diabetic People by heart Disease',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')
plt.xticks(color='k'), plt.xlabel('Diabetic',color='green',size=20),plt.yticks(()),plt.ylabel(None)
plt.show()


# In[34]:


plt.figure(figsize=(18,12))
color=['green','blue','#34495E']

plt.subplot(1,2,1)
ax2 = plt.bar(data['Smoking'].value_counts().index,data['Smoking'].value_counts().values,label='Smoking',color=color)
plt.bar_label(ax2,size=16)
plt.xlabel('Smoking'),plt.ylabel(None)
plt.title('Distribution of Smokers',color='white',backgroundcolor='k',size=18)

plt.subplot(1,2,2)
plot1=sns.countplot(data=data,x='Smoking',hue='HeartDisease',palette='viridis')
plot1.bar_label(plot1.containers[0],color='orange',size=17), plot1.bar_label(plot1.containers[1],color='brown',size=20)

plt.title('Smoking by HeartDisease',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')
plt.show()


# From the above visuals most of the patients are non smokers

# In[35]:


plt.figure(figsize=(18,12))
plt.subplot(1,2,1)
ax5=sns.countplot(data=data,x="AlcoholDrinking")
ax5.bar_label(ax5.containers[0],size=17),plt.yticks(()),plt.ylabel(None)
plt.title('Alcohol Drinking Patients',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')

plt.subplot(1,2,2)
ax6=sns.countplot(data=data,x="AlcoholDrinking",hue="HeartDisease")
ax6.bar_label(ax6.containers[0],size=16) , ax6.bar_label(ax6.containers[1],size=16,color='green')
plt.title('Alcohol Drinking Patients by HeartDisease',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.yticks(()),plt.ylabel(None)
plt.show()


# From the above visuals, most of the patients are don't drink alcohol and those that do drink, the risk of getting heart disease problems is low

# In[36]:


plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
labels=data['Stroke'].value_counts().index
vals=data['Stroke'].value_counts().values
plt.pie(vals,autopct="%1.2f%%",colors=['green','blue'],frame=True,shadow=True,wedgeprops={'edgecolor':'k'},
       textprops={'color':'white','size':15,'backgroundcolor':'black'}),plt.tight_layout(),plt.axis("equal"),plt.yticks(())
plt.xticks(()),plt.legend(labels,fontsize=15),plt.title("Heart Disease Distribution",color="white",backgroundcolor="black",fontsize=20)
plt.subplot(1,2,2)
ax7=sns.countplot(data=data,x="Stroke",hue="HeartDisease")
ax7.bar_label(ax7.containers[0],size=16) , ax7.bar_label(ax7.containers[1],size=16,color='green')
plt.title('Stroke Patients by HeartDisease',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.yticks(()),plt.ylabel(None)
plt.show()


# In[37]:


data['DiffWalking'].value_counts()
plt.figure(figsize=(18,12))
plt.subplot(1,2,1)
ax8=sns.countplot(data=data,x="DiffWalking")
ax8.bar_label(ax8.containers[0],size=17),plt.yticks(()),plt.ylabel(None)
plt.title('DiffWalking Patients',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')

plt.subplot(1,2,2)
ax9=sns.countplot(data=data,x="DiffWalking",hue="HeartDisease")
ax9.bar_label(ax9.containers[0],size=16) , ax9.bar_label(ax9.containers[1],size=16,color='green')
plt.title('DiffWalking Patients by HeartDisease',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.yticks(()),plt.ylabel(None)
plt.show()


# In[38]:


plt.figure(figsize=(10,12))
data['AgeCategory'].value_counts()
sns.barplot(x=data['AgeCategory'].value_counts().values,y=data['AgeCategory'].value_counts().index,palette="rainbow")
plt.title("AgeCategory",fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.show()


# In[39]:


plt.figure(figsize=(18,12))
data['AgeCategory'].value_counts()
ax0=sns.countplot(data=data,x="AgeCategory",hue="HeartDisease")
ax0.bar_label(ax0.containers[0],size=10) , ax0.bar_label(ax0.containers[1],size=10,color='green')
plt.title("AgeCategory",fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.show()


# from the above visual, the age category by heart disease, provides details on how the agecategory of the patients is affected by heart disease. The Heart Disease affects most people from the 50 age category

# In[40]:


plt.figure(figsize=(18,12))
plt.subplot(1,2,1)
ax=sns.countplot(data=data,x='Race',palette='viridis')
ax.bar_label(ax.containers[0],color='k',size=17)
plt.ylabel(None),plt.yticks(()),plt.xticks(rotation=45,color='k',size=8)
plt.title('Distribution of Patients by Race',color='white',backgroundcolor='k',size=18)

plt.subplot(1,2,2)
plot1=sns.countplot(data=data,x='Race',hue='HeartDisease',palette='viridis')
plt.ylabel(None),plt.yticks(()),plt.xticks(rotation=45,color='k',size=8)
plot1.bar_label(plot1.containers[0],color='k',size=10), plot1.bar_label(plot1.containers[1],color='k',size=12)

plt.title('Race by Heart Disease',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')
plt.show()


# In[41]:


data
data['Asthma'].value_counts()
plt.figure(figsize=(18,12))
plt.subplot(1,2,1)
ax8=sns.countplot(data=data,x="Asthma")
ax8.bar_label(ax8.containers[0],size=17),plt.yticks(()),plt.ylabel(None)
plt.title('Patients with Asthma',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')

plt.subplot(1,2,2)
ax9=sns.countplot(data=data,x="Asthma",hue="HeartDisease")
ax9.bar_label(ax9.containers[0],size=16) , ax9.bar_label(ax9.containers[1],size=16,color='green')
plt.title('Asthma Patients by HeartDisease',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.yticks(()),plt.ylabel(None)
plt.show()


# In[42]:


plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
labels=data['KidneyDisease'].value_counts().index
vals=data['KidneyDisease'].value_counts().values
plt.pie(vals,autopct="%1.2f%%",colors=['green','blue'],frame=True,shadow=True,wedgeprops={'edgecolor':'k'},
       textprops={'color':'white','size':15,'backgroundcolor':'black'}),plt.tight_layout(),plt.axis("equal"),plt.yticks(())
plt.xticks(()),plt.legend(labels,fontsize=15),plt.title("Distribution of Patients by KidneyDisease ",color="white",backgroundcolor="black",fontsize=20)
plt.subplot(1,2,2)
ax_1=sns.countplot(data=data,x="KidneyDisease",hue="HeartDisease")
ax_1.bar_label(ax_1.containers[0],size=16) , ax_1.bar_label(ax_1.containers[1],size=16,color='green')
plt.title('KidneyDisease by HeartDisease',fontfamily='Times New Roman',size=20,color='white',backgroundcolor='k')
plt.yticks(()),plt.ylabel(None)
plt.show()


# In[43]:


data['BMI']
plt.figure(figsize=(15,10))
sns.histplot(data=data,x="BMI",hue="HeartDisease",palette="winter_r")
plt.title('BMI levels by Heart Disease',fontfamily='Times New Roman',size=25,color='white',backgroundcolor='k')
plt.show()


# ## FEATURE SELECTION AND ENGINEERING

# In[44]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[45]:


X=data.drop("HeartDisease",axis=1)
y_labels=data['HeartDisease']


# In[46]:


encoder=LabelEncoder()
X['Smoking']=encoder.fit_transform(X['Smoking'])
X['AlcoholDrinking']=encoder.fit_transform(X['AlcoholDrinking'])
X['Stroke']=encoder.fit_transform(X['Stroke'])
X['DiffWalking']=encoder.fit_transform(X['DiffWalking'])
X['Sex']=encoder.fit_transform(X['Sex'])
X['AgeCategory']=encoder.fit_transform(X['AgeCategory'])
X['Race']=encoder.fit_transform(X['Race'])
X['Diabetic']=encoder.fit_transform(X['Diabetic'])
X['PhysicalActivity']=encoder.fit_transform(X['PhysicalActivity'])
X['GenHealth']=encoder.fit_transform(X['GenHealth'])
X['Asthma']=encoder.fit_transform(X['Asthma'])
X['KidneyDisease']=encoder.fit_transform(X['KidneyDisease'])
X['SkinCancer']=encoder.fit_transform(X['SkinCancer'])


# In[47]:


X_transformed=X
X_transformed.head()


# ### 1. Using SelectKBest with Chi square tests to find the variables that are more correlated to the target

# In[48]:


from sklearn.feature_selection import SelectKBest, chi2 


# In[49]:


features_to_select=SelectKBest(score_func=chi2,k=10)


# In[50]:


features_to_fit=features_to_select.fit(X_transformed,y_labels)


# In[51]:


print("Scores",features_to_fit.scores_)


# In[52]:


new_df=features_to_fit.transform(X_transformed)


# In[53]:


new_df


# In[54]:


feature_scores=pd.DataFrame(features_to_fit.scores_,columns=['feature_scores'])


# In[55]:


feature_columns_names=pd.DataFrame(X_transformed.columns,columns=['Feature_names'])


# In[56]:


column_relation_values=pd.concat([feature_scores,feature_columns_names],axis=1)


# In[57]:


column_relation_values


# In[58]:


column_relation_values.nlargest(10,'feature_scores')


# ### 2. Using Recursive Feature Elimination to find the more correlated variables

# In[59]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[60]:


model=LogisticRegression()
forest=RandomForestClassifier(n_estimators=200,random_state=42)


# In[61]:


rfe = RFE(model)


# In[62]:


selector=rfe.fit(X_transformed,y_labels)


# In[63]:


selected_features=pd.DataFrame(selector.support_,columns=['selected features'])


# In[64]:


feature_ranking=pd.DataFrame(selector.ranking_,columns=['ranking_features'])


# In[65]:


df=pd.concat([selected_features,feature_ranking],axis=1)


# In[66]:


df.set_index(X_transformed.columns,inplace=True)


# In[67]:


df.nlargest(16,'ranking_features')


# ### 3. Using the feature_importance_ of the classifier

# In[68]:


from sklearn.ensemble import ExtraTreesClassifier


# In[69]:


extra_trees=ExtraTreesClassifier(n_estimators=200,random_state=42)


# In[70]:


extra_trees.fit(X_transformed,y_labels)


# In[71]:


scores=extra_trees.feature_importances_


# In[72]:


series=pd.Series(np.round(scores*100,2),index=X.columns)
series


# In[73]:


data.head()


# In[74]:


selected_features=data[["HeartDisease","BMI","Smoking","AlcoholDrinking","Stroke","DiffWalking","Sex","AgeCategory","Diabetic",
                  "PhysicalActivity","KidneyDisease"]]


# The SELECTED_COLUMNS dataframe includes the variables that will be used to predict the patients risk of getting heart disease problems

# In[75]:


selected_features.head()


# In[76]:


selected_features.shape


# In[77]:


duplicates=selected_features[selected_features.duplicated()]   #the duplicates in the new dataframe


# In[78]:


duplicates.shape   #the shape of the duplicates


# In[79]:


selected_features.drop_duplicates(inplace=True)


# In[80]:


selected_features.shape


# ### Balancing the Data

# In[81]:


plt.figure(figsize=(12,10))
color=['green','blue']
selected_features['HeartDisease'].value_counts().plot.bar(color=color)
plt.title("HeartDisease",color='white',backgroundcolor='k',size=20)
plt.show()


# From the above visual, the target which is "HeartDisease" is imbalanced as the the values who HeartDisease is No is 4 times > than the values HeartDisease is Yes

# Ways of balancing the Imbalanced data:
# 1. Using SMOTE (Imblearn)
# 2. sklearn cross_val_score()
# 3. OverSampling (Imblearn)
# 4. UnderSampling (Imblearn)

# In[82]:


selected_features['HeartDisease'].value_counts()


# In[83]:


to_resample=selected_features.loc[selected_features['HeartDisease']=='Yes']
to_resample.head()


# In[84]:


our_resample=to_resample.sample(n=24723,replace=True)
our_resample.head()


# In[85]:


df=pd.concat([selected_features,our_resample])
df.head()


# ## Model Building

# In[86]:


from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score


# In[87]:


n_folds=StratifiedKFold(n_splits=10)  #Creating strata groups on the dataset.


# In[88]:


X_features=df.drop("HeartDisease",axis=1)
y_target=df["HeartDisease"]


# In[89]:


feature_columns_best=X_features.columns
feature_columns_best


# In[90]:


categorical_features=df[['Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Diabetic','PhysicalActivity',
                         'KidneyDisease']]


# In[91]:


transformer=ColumnTransformer([
("encoder", LabelEncoder(),categorical_features)],remainder='passthrough'
)


# In[92]:


encoder=LabelEncoder()
X_features['Smoking']=encoder.fit_transform(X_features['Smoking'])
X_features['AlcoholDrinking']=encoder.fit_transform(X_features['AlcoholDrinking'])
X_features['Stroke']=encoder.fit_transform(X_features['Stroke'])
X_features['DiffWalking']=encoder.fit_transform(X_features['DiffWalking'])
X_features['Sex']=encoder.fit_transform(X_features['Sex'])
X_features['AgeCategory']=encoder.fit_transform(X_features['AgeCategory'])
X_features['Diabetic']=encoder.fit_transform(X_features['Diabetic'])
X_features['PhysicalActivity']=encoder.fit_transform(X_features['PhysicalActivity'])
X_features['KidneyDisease']=encoder.fit_transform(X_features['KidneyDisease'])


# In[93]:


X_train,X_test,y_train,y_test=train_test_split(X_features,y_target,test_size=0.25,random_state=42)


# ### 1.0  KNEARESTNEIGHBORSCLASSIFIER

# In[249]:


knn_model=KNeighborsClassifier(n_neighbors=10)   #initializing the Kneighbors classifier for classification
knn_model.fit(X_train,y_train)   #training the algorithm with the training dataset.


# In[250]:


knn_model.score(X_train,y_train)   #the training accuracy of the model.


# In[251]:


#the training accuracy of the knn model using the cross validation methods that has 10 folds
cross_val_score(knn_model,X_train,y_train,cv=n_folds,scoring='accuracy').mean() 


# In[252]:


param_grid={'n_neighbors':np.arange(25,125,5),
           'weights':['uniform','distance'],
           'algorithm':['auto','brute']}

#using gridsearchcv for finding the best parameters for the model to increase the accuracy of the model.
knn_grid=GridSearchCV(knn_model,param_grid=param_grid,n_jobs=1,cv=n_folds,scoring='accuracy') 


# In[253]:


import time
get_ipython().run_line_magic('time', 'knn_grid.fit(X_train,y_train) #fitting the new knn model with the training dataset and getting the time used to run')


# In[254]:


print("The best parameter is:\t",knn_grid.best_params_) #obtaining the best parameters based on the results of the GridSearchCV
print('\n')
print("The best estimator is:\t",knn_grid.best_estimator_) #obtaining the best ESTIMATOR based On the results of the GridSearchCV
print('\n')
print("The best score is:\t",knn_grid.best_score_) #obtaining the best score of the model based on the results of the GridSearchCV


# In[255]:


knn_grid_model=knn_grid.best_estimator_
knn_grid_model.fit(X_train,y_train)


# In[256]:


#training accuracy after obtaining the best estimator and using it.
cross_val_score(knn_grid_model,X_train,y_train,cv=n_folds,scoring='accuracy').mean() 


# In[257]:


y_pred=knn_grid_model.predict(X_test)  #predicting a new dataset to validate the model efficiency.
knn_test_accuracy=accuracy_score(y_test,y_pred)   #getting the test accuracy of the model on the new dataset


# In[258]:


knn_test_accuracy


# ### 2.0 LOGISTIC REGRESSION

# In[238]:


log_model=LogisticRegression(solver='lbfgs',multi_class='auto')


# In[239]:


log_model.fit(X_train,y_train)   #the training accuracy of the model.


# In[240]:


#the training accuracy of the knn model using the cross validation methods that has 10 folds
np.round(cross_val_score(log_model,X_train,y_train,scoring="accuracy",cv=n_folds).mean(),4)


# In[241]:


param_grid={'penalty':['l1','l2'],
           'C':np.arange(0.2,1.0,0.1),
           'fit_intercept':[True,False],
           'max_iter':np.arange(100,600,100)}
log_grid=GridSearchCV(log_model,param_grid=param_grid,n_jobs=1,cv=n_folds,scoring='accuracy') 


# In[242]:


get_ipython().run_line_magic('time', 'log_grid.fit(X_train,y_train)')


# In[243]:


print("The best parameter is:\t",log_grid.best_params_) #obtaining the best parameters based on the results of the GridSearchCV
print('\n')
print("The best estimator is:\t",log_grid.best_estimator_) #obtaining the best ESTIMATOR based On the results of the GridSearchCV
print('\n')
print("The best score is:\t",log_grid.best_score_) #obtaining the best score of the model based on the results of the GridSearchCV


# In[244]:


log_grid_model=log_grid.best_estimator_
log_grid_model.fit(X_train,y_train)


# In[246]:


#training accuracy after obtaining the best estimator and using it.
np.round(cross_val_score(log_grid_model,X_train,y_train,cv=n_folds,scoring='accuracy').mean(),4) 


# In[247]:


y_pred=log_grid_model.predict(X_test)  #predicting a new dataset to validate the model efficiency.
log_test_accuracy=accuracy_score(y_test,y_pred)   #getting the test accuracy of the model on the new dataset


# In[248]:


log_test_accuracy


# ### 4.0 DECISION TREE CLASSIFIER

# In[114]:


decision_tree=DecisionTreeClassifier(random_state=42)#initializing the decuision tree classifier for classification
decision_tree.fit(X_train,y_train) #training the algorithm with the training dataset.


# In[115]:


cross_val_score(decision_tree,X_train,y_train, cv=n_folds,n_jobs=1,scoring='accuracy').mean()
 #the training accuracy of the model before obtaining the best parameters


# In[116]:


param_grid={
    'criterion': ['entropy','gini'],  #the splitting of the tree
    'max_depth': [np.arange(1,21,1),None],  #the maximum depth of the tree
    'max_features': [np.arange(11),'auto','sqrt','log2', None]}  #features to use to split the tree   #the leaf nodes
tree_grid=GridSearchCV(decision_tree,param_grid=param_grid,cv=n_folds,scoring='accuracy',n_jobs=1) 


# In[117]:


get_ipython().run_line_magic('time', 'tree_grid.fit(X_train,y_train)')


# In[118]:


print("The best parameter is:\t",tree_grid.best_params_) #obtaining the best parameters based on the results of the GridSearchCV
print('\n')
print("The best estimator is:\t",tree_grid.best_estimator_) #obtaining the best estimator based on the results of the GridSearchCV
print('\n')
print("The best score is:\t",tree_grid.best_score_) #obtaining the best score based on the results of the GridSearchCV


# In[119]:


decision_grid_model=tree_grid.best_estimator_
decision_grid_model.fit(X_train,y_train)  #fitting the best estimator with the training dataset


# In[120]:


#training accuracy after obtaining the best estimator and using it.
cross_val_score(decision_grid_model,X_train,y_train,cv=n_folds,scoring='accuracy').mean() 


# In[121]:


y_pred=decision_grid_model.predict(X_test)  #using the model to predict a new dataset and validating the model
decision_test_accuracy=accuracy_score(y_test,y_pred)  #calculating the model accuracy after validating it


# In[122]:


decision_test_accuracy


# ### 5.0 Naive Bayes classifier 

# In[233]:


from sklearn.naive_bayes import GaussianNB
gnb_clf=GaussianNB()


# In[234]:


gnb_clf.fit(X_train,y_train)


# In[235]:


cross_val_score(gnb_clf,X_train,y_train, cv=n_folds,scoring='accuracy').mean() #the training accuracy of the model


# In[236]:


y_pred=gnb_clf.predict(X_test)
Naive_bayes_accuracy=accuracy_score(y_test,y_pred)


# In[237]:


Naive_bayes_accuracy


# ### 6.0 RANDOM FOREST CLASSIFIER

# In[229]:


forest_clf=RandomForestClassifier(random_state=42)
forest_clf.fit(X_train,y_train)


# In[230]:


cross_val_score(forest_clf,X_train,y_train, cv=n_folds,scoring='accuracy').mean()


# In[231]:


y_pred=forest_clf.predict(X_test)
forest_test_accuracy=accuracy_score(y_test,y_pred)


# In[232]:


forest_test_accuracy


# ### 7.0 EXTRA TREES CLASSIFIER

# In[131]:


extra_trees=ExtraTreesClassifier(random_state=42)
extra_trees.fit(X_train,y_train)  #training the algorithm with the training dataset.


# In[132]:


cross_val_score(extra_trees,X_train,y_train, cv=n_folds,scoring='accuracy').mean() #the training accuracy of the model


# In[133]:


y_pred=extra_trees.predict(X_test)
trees_test_accuracy=accuracy_score(y_test,y_pred)


# In[134]:


trees_test_accuracy=accuracy_score(y_test,y_pred)


# In[148]:


trees_test_accuracy


# In[152]:


param_grid={
    'n_estimators':np.arange(5,95,5),
    'criterion':['gini','entropy'],
    'max_depth': [np.arange(1,8,1),None],  #the maximum depth of the tree
    'max_features': [np.arange(0,7),'auto','sqrt','log2', None]} #features to use to split the tree
extra_grid=GridSearchCV(extra_trees,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=1)


# In[153]:


get_ipython().run_line_magic('time', 'extra_grid.fit(X_train,y_train)')


# In[154]:


print("The best parameter is:\t",extra_grid.best_params_) #obtaining the best parameters based on the results of the GridSearchCV
print('\n')
print("The best estimator is:\t",extra_grid.best_estimator_) #obtaining the best estimator based on the results of the GridSearchCV
print('\n')
print("The best score is:\t",extra_grid.best_score_) #obtaining the best score based on the results of the GridSearchCV


# In[155]:


extra_grid_model=extra_grid.best_estimator_
extra_grid_model.fit(X_train,y_train)  #fitting the best estimator with the training dataset


# In[156]:


#training accuracy after obtaining the best estimator and using it.
cross_val_score(extra_grid_model,X_train,y_train,cv=n_folds,scoring='accuracy').mean() 


# In[157]:


y_pred=extra_grid_model.predict(X_test)
extra_test_accuracy=accuracy_score(y_test,y_pred)


# In[158]:


extra_test_accuracy


# ### ADABOOST CLASSIFIER

# In[187]:


adaboost_clf=AdaBoostClassifier(random_state=42)
adaboost_clf.fit(X_train,y_train)


# In[188]:


cross_val_score(adaboost_clf,X_train,y_train, cv=n_folds,scoring='accuracy').mean() #the training accuracy of the model


# In[189]:


y_pred=adaboost_clf.predict(X_test)
adaboost_test_accuracy=accuracy_score(y_test,y_pred)


# In[190]:


adaboost_test_accuracy


# ### 8.0 GradientBoostingClassifier 

# In[203]:


gradient_clf=GradientBoostingClassifier(n_estimators=5000,random_state=42)
gradient_clf.fit(X_train,y_train)


# In[204]:


cross_val_score(gradient_clf,X_train,y_train, cv=n_folds,scoring='accuracy').mean() #the training accuracy of the model


# In[205]:


y_pred=gradient_clf.predict(X_test)
gradient_test_accuracy=accuracy_score(y_test,y_pred)


# In[206]:


gradient_test_accuracy


# ## INTERPRETING AND EVALUATING THE MODEL

# In[207]:


import lime
import eli5


# #### 1.0 using Lime

# In[208]:


from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer


# In[209]:


X_train.values


# In[210]:


X_features.columns


# In[211]:


target_names=["No","Yes"]


# In[212]:


class_names=["No(0)", "Yes(1)"]


# In[213]:


explainer=LimeTabularExplainer(X_train.values,feature_names=X_features.columns,class_names=class_names,
                               discretize_continuous=True,random_state=42)


# In[214]:


decision_grid_model.predict(np.array(X_test.iloc[0]).reshape(1,-1))


# In[215]:


exp=explainer.explain_instance(X_test.iloc[0],decision_grid_model.predict_proba,num_features=10,top_labels=1)


# In[216]:


exp.show_in_notebook(show_table=True,show_all=False)


# #### 2.0 Using Eli5 

# In[217]:


eli5.show_weights(decision_grid_model,top=10,target_names=class_names)
plt.show()


# In[218]:


feature_names=X_features.columns
feature_names


# In[219]:


eli5.show_prediction(decision_grid_model,X_test.iloc[0],target_names=class_names)


# In[220]:


y_pred=decision_grid_model.predict(X_test)   #using the decision tree model to predict the unlabelled data.


# In[221]:


print(classification_report(y_test,y_pred,target_names=class_names))   #the classification report of each target class.


# In[222]:


confusion_matrix(y_test,y_pred)    #printing the confusion matrix.


# ## Saving the Models

# In[223]:


import joblib


# In[224]:


joblib.dump(extra_grid_model,"ExtraTreesModel.pkl")


# In[225]:


joblib.dump(adaboost_clf,"AdaboostClassifier.pkl")


# In[226]:


joblib.dump(gradient_clf,"GradientBoostingClassifier.pkl")


# In[ ]:





# In[ ]:




